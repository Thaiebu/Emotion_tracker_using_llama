
1. This application provides endpoints for visualizing mood and activity trends, user profile insights, and AI-based recommendations.  
2. The root endpoint serves an interactive dashboard in HTML format.  
3. Users can view mood distribution, activity time, and mood patterns for specific profiles.  
4. It includes a POST endpoint to predict mood based on user activity with optional language input.  
5. Comprehensive schema validation ensures robust error handling and structured JSON responses.  




1. **`/` Endpoint**: Serves the dashboard.  
   - **Input**: None.  
   - **Output**: HTML content.  

2. **`/visualize-mood-distribution`**: Displays a mood distribution pie chart.  
   - **Input**: None.  
   - **Output**: JSON containing a base64-encoded image of the chart.  

3. **`/visualize-activity-time`**: Shows activity time distribution as a histogram.  
   - **Input**: None.  
   - **Output**: JSON containing a base64-encoded histogram image.  

4. **`/display-ai-recommendations`**: Provides AI-generated activity recommendations.  
   - **Input**: None.  
   - **Output**: JSON array with user recommendations.  

5. **`/view-user-profile/{user_id}`**: Displays user profile and mood trends.  
   - **Input**: User ID (path parameter).  
   - **Output**: JSON with mood counts, user details, and a mood trend image.  

6. **`/predict_mood`**: Predicts mood based on activity.  
   - **Input**: JSON with activity, user name, and optional language.  
   - **Output**: JSON with predicted mood and recommendation.  

7. **Schemas**:  
   - **`MoodRequest`**: Requires `activity` and `user_name`, optionally includes `language`.  
   - **`MoodResponse`**: Includes `mood` and `recommendation`.  

8. **Validation**: Handles input errors using `HTTPValidationError`.  

9. **Content Types**: Primarily JSON responses, except for the dashboard, which returns HTML.  

10. **Error Handling**: Ensures HTTP 422 for validation issues and HTTP 500 for server errors.  
