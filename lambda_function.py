import boto3
import json

# Initialize the Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

model_Id = 'cohere.command-text-v14'

def lambda_handler(event, context):
    print('Event: ', json.dumps(event))

    # Parse the incoming request body
    requestBody = json.loads(event['body'])
    prompt = requestBody.get('prompt', '')

    # Construct the request body without the 'stop_sequence' key
    body = {
        'prompt': prompt,
        'max_tokens': 400,
        'temperature': 0.75,
        'p': 0.01,
        'k': 0,
        'return_likelihoods': 'NONE'
    }

    try:
        # Invoke the model
        bedrockResponse = bedrock.invoke_model(
            modelId=model_Id,
            body=json.dumps(body),
            accept='*/*',
            contentType='application/json'
        )

        # Process the response
        response_text = json.loads(bedrockResponse['body'].read())['generations'][0]['text']

        # Construct the API response
        apiResponse = {
            'statusCode': 200,
            'body': json.dumps({
                'prompt': prompt,
                'response': response_text
            })
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        apiResponse = {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

    return apiResponse
