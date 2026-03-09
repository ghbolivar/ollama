package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/thinking"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and images that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system messages
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, images []llm.ImageData, snapshotOffset int, _ error) {
	var system []api.Message

	// TODO: Ideally we would compute this from the projector metadata but some pieces are implementation dependent
	// Clip images are represented as 768 tokens, each an embedding
	imageNumTokens := 768

	lastMsgIdx := len(msgs) - 1
	currMsgIdx := 0

	if truncate {
		// Start with all messages and remove from the front until it fits in context
		for i := 0; i <= lastMsgIdx; i++ {
			// Collect system messages from the portion we're about to skip
			system = make([]api.Message, 0)
			for j := range i {
				if msgs[j].Role == "system" {
					system = append(system, msgs[j])
				}
			}

			p, _, err := renderPrompt(m, append(system, msgs[i:]...), tools, think)
			if err != nil {
				return "", nil, 0, err
			}

			s, err := tokenize(ctx, p)
			if err != nil {
				return "", nil, 0, err
			}

			ctxLen := len(s)
			if m.ProjectorPaths != nil {
				for _, msg := range msgs[i:] {
					ctxLen += imageNumTokens * len(msg.Images)
				}
			}

			if ctxLen <= opts.NumCtx {
				currMsgIdx = i
				break
			}

			// Must always include at least the last message
			if i == lastMsgIdx {
				currMsgIdx = lastMsgIdx
				break
			}
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating input messages which exceed context length", "truncated", len(msgs[currMsgIdx:]))
	}

	for cnt, msg := range msgs[currMsgIdx:] {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return "", nil, 0, errors.New("this model only supports one image while more than one image requested")
		}

		var prefix string
		prompt := msg.Content

		for _, i := range msg.Images {
			imgData := llm.ImageData{
				ID:   len(images),
				Data: i,
			}
			images = append(images, imgData)

			if m.Config.Renderer != "" {
				continue
			}

			imgTag := fmt.Sprintf("[img-%d]", imgData.ID)
			if !strings.Contains(prompt, "[img]") {
				prefix += imgTag
			} else {
				prompt = strings.Replace(prompt, "[img]", imgTag, 1)
			}
		}
		msgs[currMsgIdx+cnt].Content = prefix + prompt
	}

	// truncate any messages that do not fit into the context window
	p, so, err := renderPrompt(m, append(system, msgs[currMsgIdx:]...), tools, think)
	if err != nil {
		return "", nil, 0, err
	}

	return p, images, so, nil
}

func renderPrompt(m *Model, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, int, error) {
	if m.Config.Renderer != "" {
		result, err := renderers.RenderWithRenderer(m.Config.Renderer, msgs, tools, think)
		if err != nil {
			return "", 0, err
		}
		return result.Prompt, result.SnapshotOffset, nil
	}

	var b bytes.Buffer
	thinkVal := false
	thinkLevel := ""
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	}
	if err := m.Template.Execute(&b, template.Values{Messages: msgs, Tools: tools, Think: thinkVal, ThinkLevel: thinkLevel, IsThinkSet: think != nil}); err != nil {
		return "", 0, err
	}

	rendered := b.String()
	snapshotOffset := 0

	// For template-based models with thinking enabled, find the last unclosed
	// opening think tag. Prior messages have matched open/close pairs; the
	// prefill tag at the end is the only unclosed one.
	if thinkVal {
		openTag, closeTag := thinking.InferTags(m.Template.Template)
		if openTag != "" {
			if idx := strings.LastIndex(rendered, openTag); idx >= 0 {
				after := rendered[idx+len(openTag):]
				if closeTag == "" || !strings.Contains(after, closeTag) {
					snapshotOffset = idx
				}
			}
		}
	}

	return rendered, snapshotOffset, nil
}
