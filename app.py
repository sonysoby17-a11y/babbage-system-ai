import os
from flask import Flask, request, jsonify, render_template
from groq import Groq

app = Flask(__name__)

# Initialize Groq Client
# It will look for 'GROQ_API_KEY' in your Environment Variables automatically
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route('/')
def home():
    # This renders your index.html file from the templates folder
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are BabbageSystem, an advanced AI assistant."
                },
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        response = chat_completion.choices[0].message.content
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # CRITICAL FOR ZEABUR: This picks up the port assigned by the platform
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)