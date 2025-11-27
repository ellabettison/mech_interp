# Together AI Setup Guide

## Quick Setup for Cross-Model Transfer Testing

### 1. Get Together AI API Key
1. Go to https://together.ai/
2. Sign up for a free account
3. Navigate to your API keys section
4. Copy your API key

### 2. Set Environment Variable
```bash
export TOGETHER_API_KEY="your-api-key-here"
```

### 3. Run the Tests
```bash
python3 cross_model_transfer_tester.py
```

## What You Get

### Models Available:
- **Llama-2-7b**: 7B parameter Llama model
- **Llama-2-13b**: 13B parameter Llama model  
- **Mistral-7b**: 7B parameter Mistral model
- **Mixtral-8x7b**: 8x7B parameter Mixtral model

### Cost Estimate:
- **$0.20 per 1M tokens**
- **Full test run**: ~$0.008 (less than 1 cent!)

## Why Together AI?

1. **Real Models**: Access to actual Llama-2 and Mistral models
2. **Affordable**: Very reasonable pricing for testing
3. **Simple**: Just need an API key
4. **Comprehensive**: Covers the models you want to test

## Troubleshooting

### If you get 400 errors:
- Check your API key is correct
- Ensure you have sufficient credits
- Try a simpler prompt first

### If you get import errors:
- Make sure `together_ai_interface.py` is in the same directory
- Check that `requests` is installed: `pip install requests`

## Next Steps

1. **Get your API key** from Together AI
2. **Set the environment variable**
3. **Run the cross-model transfer tests**
4. **Analyze the results** across different model families

The system will automatically detect your Together AI key and test your rephrased prompts on real Llama and Mistral models! 