package gemini

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"

	"github.com/bytedance/sonic"
	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/getkin/kin-openapi/openapi3"
	"google.golang.org/api/iterator"
	"google.golang.org/genai"
)

var _ model.ToolCallingChatModel = (*ChatModel)(nil)

func NewChatModel(_ context.Context, cfg *Config) (*ChatModel, error) {
	return &ChatModel{
		cli: cfg.Client,

		model:               cfg.Model,
		maxTokens:           cfg.MaxTokens,
		temperature:         cfg.Temperature,
		topP:                cfg.TopP,
		topK:                cfg.TopK,
		responseSchema:      cfg.ResponseSchema,
		enableCodeExecution: cfg.EnableCodeExecution,
		safetySettings:      cfg.SafetySettings,
	}, nil
}

// Config contains the configuration options for the Gemini model
type Config struct {
	// Client is the Gemini API client instance
	// Required for making API calls to Gemini
	Client *genai.Client

	// Model specifies which Gemini model to use
	// Examples: "gemini-pro", "gemini-pro-vision", "gemini-1.5-flash"
	Model string

	// MaxTokens limits the maximum number of tokens in the response
	// Optional. Example: maxTokens := 100
	MaxTokens *int

	// Temperature controls randomness in responses
	// Range: [0.0, 1.0], where 0.0 is more focused and 1.0 is more creative
	// Optional. Example: temperature := float32(0.7)
	Temperature *float32

	// TopP controls diversity via nucleus sampling
	// Range: [0.0, 1.0], where 1.0 disables nucleus sampling
	// Optional. Example: topP := float32(0.95)
	TopP *float32

	// TopK controls diversity by limiting the top K tokens to sample from
	// Optional. Example: topK := int32(40)
	TopK *float32

	// ResponseSchema defines the structure for JSON responses
	// Optional. Used when you want structured output in JSON format
	ResponseSchema *genai.Schema

	// EnableCodeExecution allows the model to execute code
	// Warning: Be cautious with code execution in production
	// Optional. Default: false
	EnableCodeExecution bool

	// SafetySettings configures content filtering for different harm categories
	// Controls the model's filtering behavior for potentially harmful content
	// Optional.
	SafetySettings []*genai.SafetySetting
}

type ChatModel struct {
	cli *genai.Client

	model               string
	maxTokens           *int
	topP                *float32
	temperature         *float32
	topK                *float32
	responseSchema      *genai.Schema
	tools               []*genai.Tool
	origTools           []*schema.ToolInfo
	toolChoice          *schema.ToolChoice
	enableCodeExecution bool
	safetySettings      []*genai.SafetySetting
}

func (cm *ChatModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (message *schema.Message, err error) {
	ctx = callbacks.EnsureRunInfo(ctx, cm.GetType(), components.ComponentOfChatModel)

	if len(input) == 0 {
		return nil, fmt.Errorf("gemini input is empty")
	}

	contents, err := cm.convSchemaMessages(input)
	if err != nil {
		return nil, err
	}

	chat, conf, err := cm.initGenerativeModelChat(ctx, contents[:len(contents)-1], opts...)
	if err != nil {
		return nil, err
	}

	ctx = callbacks.OnStart(ctx, &model.CallbackInput{
		Messages: input,
		Tools:    model.GetCommonOptions(&model.Options{Tools: cm.origTools}, opts...).Tools,
		Config:   conf,
	})
	defer func() {
		if err != nil {
			callbacks.OnError(ctx, err)
		}
	}()

	parts := make([]genai.Part, 0, len(contents[len(contents)-1].Parts))
	for _, part := range contents[len(contents)-1].Parts {
		parts = append(parts, *part)
	}
	result, err := chat.SendMessage(ctx, parts...)
	if err != nil {
		return nil, fmt.Errorf("send message fail: %w", err)
	}

	message, err = cm.convResponse(result)
	if err != nil {
		return nil, fmt.Errorf("convert response fail: %w", err)
	}

	callbacks.OnEnd(ctx, cm.convCallbackOutput(message, conf))
	return message, nil
}

func (cm *ChatModel) Stream(ctx context.Context, input []*schema.Message, opts ...model.Option) (result *schema.StreamReader[*schema.Message], err error) {
	ctx = callbacks.EnsureRunInfo(ctx, cm.GetType(), components.ComponentOfChatModel)

	if len(input) == 0 {
		return nil, fmt.Errorf("gemini input is empty")
	}

	contents, err := cm.convSchemaMessages(input)
	if err != nil {
		return nil, err
	}

	chat, conf, err := cm.initGenerativeModelChat(ctx, contents[:len(contents)-1], opts...)
	if err != nil {
		return nil, err
	}

	ctx = callbacks.OnStart(ctx, &model.CallbackInput{
		Messages: input,
		Tools:    model.GetCommonOptions(&model.Options{Tools: cm.origTools}, opts...).Tools,
		Config:   conf,
	})
	defer func() {
		if err != nil {
			callbacks.OnError(ctx, err)
		}
	}()

	parts := make([]genai.Part, 0, len(contents[len(contents)-1].Parts))
	for _, part := range contents[len(contents)-1].Parts {
		parts = append(parts, *part)
	}
	resultIter := chat.SendMessageStream(ctx, parts...)

	sr, sw := schema.Pipe[*model.CallbackOutput](1)
	go func() {
		defer func() {
			panicErr := recover()

			if panicErr != nil {
				_ = sw.Send(nil, newPanicErr(panicErr, debug.Stack()))
			}
			sw.Close()
		}()
		for resp, err_ := range resultIter {
			if errors.Is(err_, iterator.Done) {
				return
			}
			if err_ != nil {
				sw.Send(nil, err_)
				return
			}
			message, err_ := cm.convResponse(resp)
			if err_ != nil {
				sw.Send(nil, err_)
				return
			}
			closed := sw.Send(cm.convCallbackOutput(message, conf), nil)
			if closed {
				return
			}
		}
	}()
	srList := sr.Copy(2)
	callbacks.OnEndWithStreamOutput(ctx, srList[0])
	return schema.StreamReaderWithConvert(srList[1], func(t *model.CallbackOutput) (*schema.Message, error) {
		return t.Message, nil
	}), nil
}

func (cm *ChatModel) WithTools(tools []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	if len(tools) == 0 {
		return nil, errors.New("no tools to bind")
	}
	gTools, err := cm.toGeminiTools(tools)
	if err != nil {
		return nil, fmt.Errorf("convert to gemini tools fail: %w", err)
	}

	tc := schema.ToolChoiceAllowed
	ncm := *cm
	ncm.toolChoice = &tc
	ncm.tools = gTools
	ncm.origTools = tools
	return &ncm, nil
}

func (cm *ChatModel) BindTools(tools []*schema.ToolInfo) error {
	if len(tools) == 0 {
		return errors.New("no tools to bind")
	}
	gTools, err := cm.toGeminiTools(tools)
	if err != nil {
		return err
	}

	cm.tools = gTools
	cm.origTools = tools
	tc := schema.ToolChoiceAllowed
	cm.toolChoice = &tc
	return nil
}

func (cm *ChatModel) BindForcedTools(tools []*schema.ToolInfo) error {
	if len(tools) == 0 {
		return errors.New("no tools to bind")
	}
	gTools, err := cm.toGeminiTools(tools)
	if err != nil {
		return err
	}

	cm.tools = gTools
	cm.origTools = tools
	tc := schema.ToolChoiceForced
	cm.toolChoice = &tc
	return nil
}

func (cm *ChatModel) initGenerativeModelChat(ctx context.Context, history []*genai.Content, opts ...model.Option) (*genai.Chat, *model.Config, error) {
	commonOptions := model.GetCommonOptions(&model.Options{
		Temperature: cm.temperature,
		MaxTokens:   cm.maxTokens,
		TopP:        cm.topP,
		Tools:       nil,
		ToolChoice:  cm.toolChoice,
	}, opts...)
	geminiOptions := model.GetImplSpecificOptions(&options{
		TopK:           cm.topK,
		ResponseSchema: cm.responseSchema,
	}, opts...)

	config := &genai.GenerateContentConfig{
		// HTTPOptions: ,
		// SystemInstruction: ,
		Temperature: commonOptions.Temperature,
		TopP:        commonOptions.TopP,
		TopK:        geminiOptions.TopK,
		// CandidateCount: ,
		// StopSequences: ,
		// ResponseLogprobs: ,
		// Logprobs: ,
		// PresencePenalty: ,
		// Seed: ,
		// ResponseMIMEType: ,
		ResponseSchema: geminiOptions.ResponseSchema,
		// RoutingConfig: ,
		// ModelSelectionConfig: ,
		// SafetySettings: ,
		// Tools: ,
		// ToolConfig: ,
		// Labels: ,
		// CachedContent: ,
		// ResponseModalities: ,
		// MediaResolution: ,
		// MediaResolution: ,
		// SpeechConfig: ,
		// AudioTimestamp: ,
		// ThinkingConfig: ,
	}
	if commonOptions.MaxTokens != nil {
		config.MaxOutputTokens = int32(*commonOptions.MaxTokens)
	}

	chat, err := cm.cli.Chats.Create(ctx, cm.model, config, history)
	if err != nil {
		return nil, nil, err
	}

	return chat, nil, nil
}

func (cm *ChatModel) toGeminiTools(tools []*schema.ToolInfo) ([]*genai.Tool, error) {
	gTools := make([]*genai.Tool, len(tools))
	for i, tool := range tools {
		funcDecl := &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Desc,
		}

		openSchema, err := tool.ToOpenAPIV3()
		if err != nil {
			return nil, fmt.Errorf("get open schema fail: %w", err)
		}
		funcDecl.Parameters, err = cm.convOpenSchema(openSchema)
		if err != nil {
			return nil, fmt.Errorf("convert open schema fail: %w", err)
		}

		gTools[i] = &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{funcDecl},
		}
	}

	return gTools, nil
}

func (cm *ChatModel) convOpenSchema(schema *openapi3.Schema) (*genai.Schema, error) {
	if schema == nil {
		return nil, nil
	}
	var err error

	result := &genai.Schema{
		Format:      schema.Format,
		Description: schema.Description,
		Nullable:    genai.Ptr(schema.Nullable),
	}

	switch schema.Type {
	case openapi3.TypeObject:
		result.Type = genai.TypeObject
		if schema.Properties != nil {
			properties := make(map[string]*genai.Schema)
			for name, prop := range schema.Properties {
				if prop == nil || prop.Value == nil {
					continue
				}
				properties[name], err = cm.convOpenSchema(prop.Value)
				if err != nil {
					return nil, err
				}
			}
			result.Properties = properties
		}
		if schema.Required != nil {
			result.Required = schema.Required
		}

	case openapi3.TypeArray:
		result.Type = genai.TypeArray
		if schema.Items != nil && schema.Items.Value != nil {
			result.Items, err = cm.convOpenSchema(schema.Items.Value)
			if err != nil {
				return nil, err
			}
		}

	case openapi3.TypeString:
		result.Type = genai.TypeString
		if schema.Enum != nil {
			enums := make([]string, 0, len(schema.Enum))
			for _, e := range schema.Enum {
				if str, ok := e.(string); ok {
					enums = append(enums, str)
				} else {
					return nil, fmt.Errorf("enum value must be a string, schema: %+v", schema)
				}
			}
			result.Enum = enums
		}

	case openapi3.TypeNumber:
		result.Type = genai.TypeNumber
	case openapi3.TypeInteger:
		result.Type = genai.TypeInteger
	case openapi3.TypeBoolean:
		result.Type = genai.TypeBoolean
	default:
		result.Type = genai.TypeUnspecified
	}

	return result, nil
}

func (cm *ChatModel) convSchemaMessages(messages []*schema.Message) ([]*genai.Content, error) {
	result := make([]*genai.Content, len(messages))
	for i, message := range messages {
		content, err := cm.convSchemaMessage(message)
		if err != nil {
			return nil, fmt.Errorf("convert schema message fail: %w", err)
		}
		result[i] = content
	}
	return result, nil
}

func (cm *ChatModel) convSchemaMessage(message *schema.Message) (*genai.Content, error) {
	if message == nil {
		return nil, nil
	}

	content := &genai.Content{
		Role: toGeminiRole(message.Role),
	}

	if message.ToolCalls != nil {
		for _, call := range message.ToolCalls {
			args := make(map[string]any)
			err := sonic.UnmarshalString(call.Function.Arguments, &args)
			if err != nil {
				return nil, fmt.Errorf("unmarshal schema tool call arguments to map[string]any fail: %w", err)
			}
			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					Name: call.Function.Name,
					Args: args,
				},
			})
		}
	}

	if message.Role == schema.Tool {
		response := make(map[string]any)
		err := sonic.UnmarshalString(message.Content, &response)
		if err != nil {
			return nil, fmt.Errorf("unmarshal schema tool call response to map[string]any fail: %w", err)
		}
		content.Parts = append(content.Parts, &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				Name:     message.ToolCallID,
				Response: response,
			},
		})
	} else {
		if message.Content != "" {
			content.Parts = append(content.Parts, &genai.Part{
				Text: message.Content,
			})
		}
		content.Parts = append(content.Parts, cm.convMedia(message.MultiContent)...)
	}
	return content, nil
}

func (cm *ChatModel) convMedia(contents []schema.ChatMessagePart) []*genai.Part {
	result := make([]*genai.Part, 0, len(contents))
	for _, content := range contents {
		switch content.Type {
		case schema.ChatMessagePartTypeText:
			result = append(result, &genai.Part{
				Text: content.Text,
			})
		case schema.ChatMessagePartTypeImageURL:
			if content.ImageURL != nil {
				result = append(result, &genai.Part{
					FileData: &genai.FileData{
						MIMEType: content.ImageURL.MIMEType,
						FileURI:  content.ImageURL.URI,
					},
				})
			}
		case schema.ChatMessagePartTypeAudioURL:
			if content.AudioURL != nil {
				result = append(result, &genai.Part{
					FileData: &genai.FileData{
						MIMEType: content.AudioURL.MIMEType,
						FileURI:  content.AudioURL.URI,
					},
				})
			}
		case schema.ChatMessagePartTypeVideoURL:
			if content.VideoURL != nil {
				result = append(result, &genai.Part{
					FileData: &genai.FileData{
						MIMEType: content.VideoURL.MIMEType,
						FileURI:  content.VideoURL.URI,
					},
				})
			}
		case schema.ChatMessagePartTypeFileURL:
			if content.FileURL != nil {
				result = append(result, &genai.Part{
					FileData: &genai.FileData{
						MIMEType: content.FileURL.MIMEType,
						FileURI:  content.FileURL.URI,
					},
				})
			}
		}
	}
	return result
}

func (cm *ChatModel) convResponse(resp *genai.GenerateContentResponse) (*schema.Message, error) {
	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("gemini result is empty")
	}

	message, err := cm.convCandidate(resp.Candidates[0])
	if err != nil {
		return nil, fmt.Errorf("convert candidate fail: %w", err)
	}

	if resp.UsageMetadata != nil {
		if message.ResponseMeta == nil {
			message.ResponseMeta = &schema.ResponseMeta{}
		}
		message.ResponseMeta.Usage = &schema.TokenUsage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(resp.UsageMetadata.TotalTokenCount),
		}
	}
	return message, nil
}

func (cm *ChatModel) convCandidate(candidate *genai.Candidate) (*schema.Message, error) {
	result := &schema.Message{}
	result.ResponseMeta = &schema.ResponseMeta{
		FinishReason: string(candidate.FinishReason),
	}
	if candidate.Content != nil {
		if candidate.Content.Role == roleModel {
			result.Role = schema.Assistant
		} else {
			result.Role = schema.User
		}

		var texts []string
		for _, part := range candidate.Content.Parts {
			if part.Text != "" {
				if part.Thought {
					continue
				}
				texts = append(texts, part.Text)
			} else if part.FunctionCall != nil {
				fc, err := convFC(part.FunctionCall)
				if err != nil {
					return nil, err
				}
				result.ToolCalls = append(result.ToolCalls, *fc)
			} else if part.CodeExecutionResult != nil {
				texts = append(texts, part.CodeExecutionResult.Output)
			} else if part.ExecutableCode != nil {
				texts = append(texts, part.ExecutableCode.Code)
			}
		}
		if len(texts) == 1 {
			result.Content = texts[0]
		} else if len(texts) > 1 {
			for _, text := range texts {
				result.MultiContent = append(result.MultiContent, schema.ChatMessagePart{
					Type: schema.ChatMessagePartTypeText,
					Text: text,
				})
			}
		}
	}
	return result, nil
}

func convFC(tp *genai.FunctionCall) (*schema.ToolCall, error) {
	args, err := sonic.MarshalString(tp.Args)
	if err != nil {
		return nil, fmt.Errorf("marshal gemini tool call arguments fail: %w", err)
	}
	return &schema.ToolCall{
		ID: tp.Name,
		Function: schema.FunctionCall{
			Name:      tp.Name,
			Arguments: args,
		},
	}, nil
}

func (cm *ChatModel) convCallbackOutput(message *schema.Message, conf *model.Config) *model.CallbackOutput {
	callbackOutput := &model.CallbackOutput{
		Message: message,
		Config:  conf,
	}
	if message.ResponseMeta != nil && message.ResponseMeta.Usage != nil {
		callbackOutput.TokenUsage = &model.TokenUsage{
			PromptTokens:     message.ResponseMeta.Usage.PromptTokens,
			CompletionTokens: message.ResponseMeta.Usage.CompletionTokens,
			TotalTokens:      message.ResponseMeta.Usage.TotalTokens,
		}
	}
	return callbackOutput
}

const (
	roleModel = "model"
	roleUser  = "user"
)

func toGeminiRole(role schema.RoleType) string {
	if role == schema.Assistant {
		return roleModel
	}
	return roleUser
}

const typ = "Gemini"

func (cm *ChatModel) GetType() string {
	return typ
}

type panicErr struct {
	info  any
	stack []byte
}

func (p *panicErr) Error() string {
	return fmt.Sprintf("panic error: %v, \nstack: %s", p.info, string(p.stack))
}

func newPanicErr(info any, stack []byte) error {
	return &panicErr{
		info:  info,
		stack: stack,
	}
}
