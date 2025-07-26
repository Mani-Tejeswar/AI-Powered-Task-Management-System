from flask import Flask, render_template, request
from Models import preprocess_text, tfidf_vectorizer, final_classification_model, category_encoder, final_priority_model, priority_encoder, user_workload_df
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        task_description = request.form['task_description']
        user_workload = int(request.form['user_workload'])

        # Preprocess and vectorize
        processed = preprocess_text(task_description)
        text_vector = tfidf_vectorizer.transform([processed])
        workload_input = np.array([[user_workload]])
        combined_vector = hstack([text_vector, workload_input])

        # Predict category
        predicted_category_encoded = final_classification_model.predict(text_vector)[0]
        predicted_category = category_encoder.inverse_transform([predicted_category_encoded])[0]

        # Predict priority
        predicted_priority_encoded = final_priority_model.predict(combined_vector)[0]
        predicted_priority = priority_encoder.inverse_transform([predicted_priority_encoded])[0]

        # Recommend user (lowest workload)
        avg_workload = user_workload_df.groupby('assigned_user')['user_workload'].mean()
        best_user = avg_workload.idxmin()

        result = {
            'category': predicted_category,
            'priority': predicted_priority,
            'user': best_user
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)