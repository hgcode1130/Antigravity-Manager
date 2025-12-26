// OpenAI → Gemini 请求转换
use super::models::*;
use serde_json::{json, Value};
use super::streaming::get_thought_signature;

pub fn transform_openai_request(request: &OpenAIRequest, project_id: &str, mapped_model: &str) -> Value {
    // Resolve grounding config
    let config = crate::proxy::mappers::common_utils::resolve_request_config(&request.model, mapped_model);

    tracing::info!("[Debug] OpenAI Request: original='{}', mapped='{}', type='{}', has_image_config={}", 
        request.model, mapped_model, config.request_type, config.image_config.is_some());
    
    // 构建 Gemini contents 和 systemInstruction
    let mut contents = Vec::new();
    let mut system_instruction = None;

    // Pre-scan to map tool_call_id to function name
    let mut tool_id_to_name = std::collections::HashMap::new();
    for msg in &request.messages {
        if let Some(tool_calls) = &msg.tool_calls {
            if let Some(calls_arr) = tool_calls.as_array() {
                for call in calls_arr {
                   if let (Some(id), Some(func)) = (call.get("id").and_then(|v| v.as_str()), call.get("function")) {
                       if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                           let final_name = if name == "local_shell_call" { "shell" } else { name };
                           tool_id_to_name.insert(id.to_string(), final_name.to_string());
                       }
                   }
                }
            }
        }
    }

    // 从全局存储获取 thoughtSignature（不再从文本中提取）
    let global_thought_sig = get_thought_signature();
    if global_thought_sig.is_some() {
        tracing::info!("从全局存储获取到 thoughtSignature (长度: {})", global_thought_sig.as_ref().unwrap().len());
    }

    for msg in &request.messages {
        if msg.role == "system" {
            let content_str = msg.content.as_ref().map(|v| {
                if v.is_string() { v.as_str().unwrap().to_string() }
                else { v.to_string() }
            }).unwrap_or_default();
            
            system_instruction = Some(json!({
                "parts": [{"text": format!("{}\n\n[SYSTEM NOTE: You are a coding agent. You MUST use the provided 'shell' tool to perform ANY filesystem operations (reading, writing, creating files). Do not output JSON code blocks for tool execution; invoke the functions directly. To create a file, use the 'shell' tool with 'New-Item' or 'Set-Content' (Powershell). NEVER simulate/hallucinate actions in text without calling the tool first.]", content_str)}]
            }));
            continue;
        }

        let role = match msg.role.as_str() {
            "assistant" => "model",
            "tool" | "function" => "user", // Gemini often expects function responses as 'user' role
            _ => "user",
        };

        let mut parts = Vec::new();

        if let Some(tool_calls) = &msg.tool_calls {
            let mut has_content_been_used = false;
            let original_content = msg.content.as_ref().map(|v| {
                if v.is_string() { v.as_str().unwrap().to_string() }
                else { v.to_string() }
            }).unwrap_or_default();

            // 注：不再需要从文本中提取签名，直接使用全局存储的签名
            let clean_content = original_content.clone();

            if let Some(calls_arr) = tool_calls.as_array() {
                for (index, call) in calls_arr.iter().enumerate() {
                    // INJECT THOUGHT before EACH function call
                    // Priority: 1. Original Content (only for first call) 2. Dummy Thought (if Gemini-3)
                    if index == 0 && !clean_content.is_empty() {
                         parts.push(json!({"text": clean_content}));
                         has_content_been_used = true;
                    } else if mapped_model.contains("gemini-3") {
                         parts.push(json!({"text": "Thinking Process: Determining necessary tool actions."}));
                    }

                    if let Some(func) = call.get("function") {
                        let raw_name = func.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let name = if raw_name == "local_shell_call" { "shell" } else { raw_name };
                        
                        let args_str = func.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
                        let args: Value = serde_json::from_str(args_str).unwrap_or_else(|e| {
                            tracing::error!("Failed to parse arguments: {}, error: {}", args_str, e);
                            json!({})
                        });
                        tracing::debug!("Function {} args: {:?}", name, args);
                        
                        // 构建 functionCall part，如果有签名则注入
                        // 根据官方文档：thoughtSignature 应与 functionCall 并列在 part 级别
                        let mut func_call_part = json!({
                            "functionCall": {
                                "name": name,
                                "args": args
                            }
                        });
                        // 仅第一个 functionCall 需要签名（签名与 functionCall 并列，不是嵌套在内部）
                        if index == 0 {
                            // 使用全局存储的签名
                            if let Some(ref sig) = global_thought_sig {
                                // 正确位置：与 functionCall 并列放在 part 根级别
                                func_call_part["thoughtSignature"] = json!(sig);
                                tracing::info!("注入 thoughtSignature 到 part 级别 (长度: {})", sig.len());
                            } else {
                                tracing::warn!("无法找到 thoughtSignature，可能导致 Gemini 3 模型报错");
                            }
                        }
                        parts.push(func_call_part);
                    }
                }
            }
        } else if msg.role == "tool" || msg.role == "function" {
            // Function Response
            let raw_name = msg.name.as_deref().unwrap_or("unknown");
            let mut name = if raw_name == "local_shell_call" { "shell" } else { raw_name };
            
            // Try to resolve name from tool_call_id
            if let Some(tid) = &msg.tool_call_id {
                if let Some(resolved) = tool_id_to_name.get(tid) {
                    name = resolved;
                }
            }
            
            tracing::info!("DEBUG: Mapping Function Response: ID={:?}, Name={}, Resolved={}", msg.tool_call_id, raw_name, name);

            let content_str = msg.content.as_ref().map(|v| {
                if v.is_string() { v.as_str().unwrap().to_string() }
                else { v.to_string() }
            }).unwrap_or_default();
            
            parts.push(json!({
                "functionResponse": {
                    "name": name,
                    "id": msg.tool_call_id.as_deref().unwrap_or("unknown"),
                    "response": { "content": content_str }
                }
            }));
        } else {
            // Regular Text Content - 支持文本和图片
            if let Some(content) = &msg.content {
                // 检查是否是数组格式 (OpenAI 多模态消息)
                if let Some(content_arr) = content.as_array() {
                    for item in content_arr {
                        if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                            match item_type {
                                "text" => {
                                    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                                        if !text.is_empty() {
                                            if role == "user" {
                                                let reminder = "\n\n(SYSTEM REMINDER: You MUST use the 'shell' tool to perform this action. Do not simply state it is done.)";
                                                parts.push(json!({ "text": format!("{}{}", text, reminder) }));
                                            } else {
                                                parts.push(json!({ "text": text }));
                                            }
                                        }
                                    }
                                }
                                "image_url" => {
                                    // OpenAI 格式: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                                    if let Some(img_obj) = item.get("image_url") {
                                        if let Some(url) = img_obj.get("url").and_then(|v| v.as_str()) {
                                            // 解析 data URL: data:image/png;base64,xxxxx
                                            if url.starts_with("data:") {
                                                if let Some(comma_pos) = url.find(',') {
                                                    let header = &url[5..comma_pos]; // 跳过 "data:"
                                                    let base64_data = &url[comma_pos + 1..];
                                                    
                                                    // 解析 MIME 类型
                                                    let mime_type = if let Some(semi_pos) = header.find(';') {
                                                        &header[..semi_pos]
                                                    } else {
                                                        header
                                                    };
                                                    
                                                    tracing::info!("[OpenAI→Gemini] 转换图片: MIME={}, 数据长度={}", mime_type, base64_data.len());
                                                    
                                                    // 转换为 Gemini inlineData 格式
                                                    parts.push(json!({
                                                        "inlineData": {
                                                            "mimeType": mime_type,
                                                            "data": base64_data
                                                        }
                                                    }));
                                                }
                                            } else if url.starts_with("http") {
                                                // 网络图片 URL - 使用 fileData 格式
                                                tracing::info!("[OpenAI→Gemini] 网络图片 URL: {}", url);
                                                parts.push(json!({
                                                    "fileData": {
                                                        "fileUri": url,
                                                        "mimeType": "image/jpeg"
                                                    }
                                                }));
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    tracing::warn!("[OpenAI→Gemini] 未知内容类型: {}", item_type);
                                }
                            }
                        }
                    }
                } else if content.is_string() {
                    // 简单字符串格式
                    let content_str = content.as_str().unwrap();
                    if !content_str.is_empty() {
                        if role == "user" {
                            let reminder = "\n\n(SYSTEM REMINDER: You MUST use the 'shell' tool to perform this action. Do not simply state it is done.)";
                            parts.push(json!({ "text": format!("{}{}", content_str, reminder) }));
                        } else {
                            parts.push(json!({ "text": content_str }));
                        }
                    }
                }
            }
        }

        if !parts.is_empty() {
            contents.push(json!({
                "role": role,
                "parts": parts
            }));
        }
    }

    // 构建请求体
    let mut inner_request = json!({
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": request.max_tokens.unwrap_or(8192),
            "temperature": request.temperature.unwrap_or(1.0),
            "topP": request.top_p.unwrap_or(1.0), 
        },
        "safetySettings": [
            { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF" },
            { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF" },
            { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF" },
            { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF" },
        ]
    });

    if let Some(si) = system_instruction {
        inner_request.as_object_mut().unwrap().insert("systemInstruction".to_string(), si);
    }

    // Map Tools
    if let Some(tools) = &request.tools {
        let mut gemini_tools = Vec::new();
        let mut function_declarations = Vec::new();
        
        for tool in tools {
            if let Some(tool_type) = tool.get("type").and_then(|v| v.as_str()) {
                if tool_type == "function" {
                    // Try to get "function" object (OpenAI standard) OR use tool itself (Codex flat format)
                    let mut gemini_func = if let Some(function) = tool.get("function") {
                        function.clone()
                    } else {
                        // Flat format: the tool itself is the function definition, but we need to remove 'type'
                        let mut func = tool.clone();
                         if let Some(obj) = func.as_object_mut() {
                            obj.remove("type"); // Remove "type": "function" from function definition
                            obj.remove("strict");
                            obj.remove("additionalProperties");
                        }
                        func
                    };

                    // Map local_shell_call to shell for definition
                    if let Some(name) = gemini_func.get("name").and_then(|v| v.as_str()) {
                        if name == "local_shell_call" {
                            if let Some(obj) = gemini_func.as_object_mut() {
                                obj.insert("name".to_string(), json!("shell"));
                            }
                        }
                    }

                    // Recursive mapping of types to uppercase
                    if let Some(params) = gemini_func.get_mut("parameters") {
                        // Gemini requires top-level parameters to be an OBJECT
                        if let Some(params_obj) = params.as_object_mut() {
                            if !params_obj.contains_key("type") {
                                params_obj.insert("type".to_string(), json!("OBJECT"));
                            }
                        }
                        map_json_schema_to_gemini(params);
                    }
                    function_declarations.push(gemini_func);
                }
            }
        }
        



        // Ensure no empty function declarations
        if !function_declarations.is_empty() {
             gemini_tools.push(json!({
                "function_declarations": function_declarations
            }));
            inner_request.as_object_mut().unwrap().insert("tools".to_string(), json!(gemini_tools));
        }
    }
    
    // Inject googleSearch tool if needed

    if config.inject_google_search {
        crate::proxy::mappers::common_utils::inject_google_search_tool(&mut inner_request);
    }

    // Inject imageConfig if present (for image generation models)
    if let Some(image_config) = config.image_config {
         if let Some(obj) = inner_request.as_object_mut() {
             // 1. Remove tools (image generation does not support tools)
             obj.remove("tools");
             
             // 2. Remove systemInstruction (image generation does not support system prompts)
             obj.remove("systemInstruction");

             // 3. Clean generationConfig (remove thinkingConfig, responseMimeType, responseModalities etc.)
             let gen_config = obj.entry("generationConfig").or_insert_with(|| json!({}));
             if let Some(gen_obj) = gen_config.as_object_mut() {
                 gen_obj.remove("thinkingConfig");
                 gen_obj.remove("responseMimeType"); 
                 gen_obj.remove("responseModalities");
                 gen_obj.insert("imageConfig".to_string(), image_config);
             }
         }
    }

    let final_request = json!({
        "project": project_id,
        "requestId": format!("openai-{}", uuid::Uuid::new_v4()),
        "request": inner_request,
        "model": config.final_model,
        "userAgent": "antigravity-openai", 
        "requestType": config.request_type
    });
    
    tracing::info!("[Debug] Final Gemini Request Body: {}", serde_json::to_string(&final_request).unwrap_or_default());
    
    tracing::info!("Final Gemini Request Body: {}", serde_json::to_string_pretty(&final_request).unwrap_or_default());
    final_request
}

fn map_json_schema_to_gemini(value: &mut Value) {
    if let Some(obj) = value.as_object_mut() {
        // Whitelist filtering: Remove all keys NOT in this list
        // This effectively removes "strict", "additionalProperties", "title", "default", etc.
        let allowed_keys = ["type", "description", "properties", "required", "items", "enum", "format", "nullable"];
        obj.retain(|k, _| allowed_keys.contains(&k.as_str()));

        // Upper case type
        let type_str = obj.get("type").and_then(|t| t.as_str()).map(|s| s.to_string());
        if let Some(s) = type_str {
            obj.insert("type".to_string(), json!(s.to_uppercase()));
        }
        
        if let Some(properties) = obj.get_mut("properties") {
            if let Some(props_obj) = properties.as_object_mut() {
                for (_, prop_val) in props_obj {
                    map_json_schema_to_gemini(prop_val);
                }
            }
        }
        
        if let Some(items) = obj.get_mut("items") {
             map_json_schema_to_gemini(items);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_openai_request() {
        let req = OpenAIRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: Some(json!("Hello")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            instructions: None,
            input: None,
        };

        let result = transform_openai_request(&req, "test-project", "gemini-1.5-pro-latest");
        assert_eq!(result["project"], "test-project");
        assert!(result["requestId"].as_str().unwrap().starts_with("openai-"));
    }
}
