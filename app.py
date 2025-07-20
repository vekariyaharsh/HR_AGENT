from flask import Flask
import slack
import os 



app = Flask(__name__)

app.register_blueprint(slack.slack_bot)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port, debug=True)