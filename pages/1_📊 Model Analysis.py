import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, skew, kurtosis
from clean_data import load_player_stats, add_target_variable
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from openai import OpenAI

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="ML Model Analysis",
    page_icon="📊"
)

player_stats, team_stats = load_player_stats()
target_variables = team_stats.columns
player_variables = player_stats.columns

st.title("Configure Your Machine Learning Model")

st.subheader("Step 1: Select a team-level target variable")

# Filter out "Games" and "Team Name" from selectbox options (but keep them in dataframe)
target_variables_filtered = [col for col in target_variables if col not in ['Games', 'Team Name']]

# Save the selected target variable
selected_target = st.selectbox("Select a target variable", target_variables_filtered)

trigger = st.button("Select", key="button_1")

if trigger:
    player_stats_with_target = add_target_variable(player_stats.copy(), team_stats, selected_target)
    st.session_state['player_stats_with_target'] = player_stats_with_target
    st.write(player_stats_with_target)

st.divider()

st.subheader("Step 2: Select a Model")

# Create visual separation between linear and non-linear models
st.markdown("**📈 Linear Models:**")
linear_options = ['LassoCV', 'ElasticNetCV', 'RidgeCV']
st.write('LassoCV - ', 'ElasticNetCV - ', 'RidgeCV')
st.markdown("**📊 Non-Linear Models:**")
nonlinear_options = ['GradientBoosting', 'Random Forest', 'XGBoost']
st.write('GradientBoosting - ', 'Random Forest - ', 'XGBoost')

# Combine options maintaining order
all_options = linear_options + nonlinear_options

# Single radio button with all options
model_choice = st.radio(
    "Select a model",
    all_options,
    horizontal=True,
    key="model_selection"
)

# Show different inputs based on model choice
if model_choice == 'LassoCV':
    cv_folds = st.number_input("Number of CV folds", min_value=2, max_value=10, value=5, step=1, key="lasso_cv")

elif model_choice == 'RidgeCV':
    col1, col2 = st.columns(2)
    with col1:  
        cv_folds = st.number_input("Number of CV folds", min_value=2, max_value=10, value=5, step=1, key="ridge_cv")
    with col2:
        alpha = st.number_input("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="ridge_alpha")

elif model_choice == 'ElasticNetCV':
    col1, col2 = st.columns(2)
    with col1:
        cv_folds = st.number_input("Number of CV folds", min_value=2, max_value=10, value=5, step=1, key="elastic_cv")
    with col2:
        max_iter = st.number_input("Max iterations", min_value=100, max_value=5000, value=2000, step=100, key="elastic_maxiter")
    
elif model_choice == 'GradientBoosting':
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_estimators = st.number_input("Number of estimators", min_value=50, max_value=1000, value=300, step=50, key="gb_n_est")
    with col2:
        learning_rate = st.number_input("Learning rate", min_value=0.01, max_value=1.0, value=0.05, step=0.01, key="gb_lr")
    with col3:
        max_depth = st.number_input("Max depth", min_value=1, max_value=10, value=2, step=1, key="gb_depth")
    with col4:
        subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key="gb_subsample")

elif model_choice == 'Random Forest':
    col1, col2 = st.columns(2)
    with col1:
        rf_n_estimators = st.number_input("Number of estimators", min_value=50, max_value=1000, value=100, step=50, key="rf_n_est")
    with col2:
        rf_max_depth = st.number_input("Max depth", min_value=1, max_value=20, value=10, step=1, key="rf_depth")

elif model_choice == 'XGBoost':
    col1, col2, col3 = st.columns(3)
    with col1:
        xgb_n_estimators = st.number_input("Number of estimators", min_value=50, max_value=1000, value=100, step=50, key="xgb_n_est")
    with col2:
        xgb_learning_rate = st.number_input("Learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="xgb_lr")
    with col3:
        xgb_max_depth = st.number_input("Max depth", min_value=1, max_value=10, value=3, step=1, key="xgb_depth")

trigger_4 = st.button("Run", key="button_4")
if trigger_4:
    # Make sure step 1 has been run
    if 'player_stats_with_target' not in st.session_state:
        st.error("Please run Step 1 first to generate 'player_stats_with_target'.")
    else:
        player_stats_with_target = st.session_state['player_stats_with_target']
        
        if model_choice == 'LassoCV':
            target_col_name = f'{selected_target}_target'
            X = player_stats_with_target.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col_name], errors='ignore')
            exclude_cols = ['Name', 'Team', 'Position']
            X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
            y = player_stats_with_target[target_col_name]
            model = LassoCV(cv=int(cv_folds), random_state=12)
            model.fit(X, y)
            st.session_state['model'] = model
            st.session_state['model_type'] = 'LassoCV'
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.write("Best alpha:", model.alpha_)
            st.write("R2:", model.score(X, y))

        elif model_choice == 'ElasticNetCV':
            target_col_name = f'{selected_target}_target'
            X = player_stats_with_target.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col_name], errors='ignore')
            exclude_cols = ['Name', 'Team', 'Position']
            X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
            y = player_stats_with_target[target_col_name]
            model = make_pipeline(
                StandardScaler(),
                ElasticNetCV(
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    cv=int(cv_folds),
                    max_iter=int(max_iter),
                    random_state=12,
                    n_jobs=-1  # Use all CPU cores for faster cross-validation
                )
            )
            model.fit(X, y)
            st.session_state['model'] = model
            st.session_state['model_type'] = 'ElasticNetCV'
            st.session_state['X'] = X
            st.session_state['y'] = y
            enet = model.named_steps['elasticnetcv']
            st.write("Best alpha:", enet.alpha_)
            st.write("Best l1_ratio:", enet.l1_ratio_)
            st.write("R2:", model.score(X, y))
            st.write("Number of features selected:", sum(enet.coef_ != 0))
    
        elif model_choice == 'RidgeCV':
            target_col_name = f'{selected_target}_target'
            X = player_stats_with_target.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col_name], errors='ignore')
            exclude_cols = ['Name', 'Team', 'Position']
            X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
            y = player_stats_with_target[target_col_name]
            model = RidgeCV(cv=int(cv_folds), alpha=float(alpha), random_state=12)
            model.fit(X, y)
            st.session_state['model'] = model
            st.session_state['model_type'] = 'RidgeCV'
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.write("Best alpha:", model.alpha_)
            st.write("R2:", model.score(X, y))
            st.write("Number of features selected:", sum(model.coef_ != 0))

        elif model_choice == 'GradientBoosting':
            target_col_name = f'{selected_target}_target'
            X = player_stats_with_target.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col_name], errors='ignore')
            exclude_cols = ['Name', 'Team', 'Position']
            X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
            y = player_stats_with_target[target_col_name]           
            model = GradientBoostingRegressor(
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                max_depth=int(max_depth),
                subsample=float(subsample),
                random_state=12
            )
            model.fit(X, y)
            st.session_state['model'] = model
            st.session_state['model_type'] = 'GradientBoosting'
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.write("R2:", model.score(X, y))

        elif model_choice == 'Random Forest':
            target_col_name = f'{selected_target}_target'
            X = player_stats_with_target.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col_name], errors='ignore')
            exclude_cols = ['Name', 'Team', 'Position']
            X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
            y = player_stats_with_target[target_col_name]
            model = RandomForestRegressor(
                n_estimators=int(rf_n_estimators),
                max_depth=int(rf_max_depth),
                random_state=12
            )
            model.fit(X, y)
            st.session_state['model'] = model
            st.session_state['model_type'] = 'Random Forest'
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.write("R2:", model.score(X, y))

        elif model_choice == 'XGBoost':
            target_col_name = f'{selected_target}_target'
            X = player_stats_with_target.select_dtypes(include=['int64', 'float64']).drop(columns=[target_col_name], errors='ignore')
            exclude_cols = ['Name', 'Team', 'Position']
            X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
            y = player_stats_with_target[target_col_name]
            model = XGBRegressor(
                n_estimators=int(xgb_n_estimators),
                learning_rate=float(xgb_learning_rate),
                max_depth=int(xgb_max_depth),
                random_state=12
            )
            model.fit(X, y)
            st.session_state['model'] = model
            st.session_state['model_type'] = 'XGBoost'
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.write("R2:", model.score(X, y))

st.divider()
st.subheader("Step 3: Graph The Model's performance")

# Function to display graphs (reusable)
def display_graphs():
    model = st.session_state['model']
    X = st.session_state['X']
    y = st.session_state['y']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    predictions = model.predict(X)
    ax.scatter(y, predictions, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title("Actual vs Predicted", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.write("Each point represents a sample: the x-axis is the actual value and the y-axis is the predicted value. "
            "If the model were perfect, all points would fall on the diagonal line. "
            "The closer the points are to that line, the more accurate the model is")

    # Calculate metrics for graph context
    correlation, _ = pearsonr(y, predictions)
    mae = np.mean(np.abs(y - predictions))
    rmse = np.sqrt(np.mean((y - predictions)**2))
    
    residuals = y - predictions
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    residual_skewness = skew(residuals)
    residual_kurtosis = kurtosis(residuals)
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(predictions, residuals, alpha=0.5)
    ax2.axhline(0, linestyle="--", color='r', lw=2)
    ax2.set_xlabel("Predicted", fontsize=12)
    ax2.set_ylabel("Residual (Actual - Predicted)", fontsize=12)
    ax2.set_title("Residuals vs Predicted", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    st.write("Each point represents the residual (actual − predicted) plotted against the predicted value. "
            "A good model will show points randomly scattered around zero with no clear pattern. "
            "Patterns or curves indicate the model is missing structure in the data or is biased.")


    # Feature importance/coefficients plot - different for each model type
    model_type = st.session_state.get('model_type', 'Unknown')
    y_label = None  # Initialize y_label
    top_features_list = []
    top_features_values = []

    if model_type == 'LassoCV':
        # LassoCV has coef_ directly
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coef": model.coef_
        }).sort_values("coef", key=abs, ascending=False).head(15)
        y_label = "Coefficient"
        title = "Top 15 Features by |Coefficient| (LassoCV)"
        
    elif model_type == 'ElasticNetCV':
        # ElasticNetCV is in a pipeline, need to access nested estimator
        enet = model.named_steps['elasticnetcv']
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coef": enet.coef_
        }).sort_values("coef", key=abs, ascending=False).head(15)
        y_label = "Coefficient"
        title = "Top 15 Features by |Coefficient| (ElasticNetCV)"
        
    elif model_type == 'RidgeCV':
        # RidgeCV has coef_ directly
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coef": model.coef_
        }).sort_values("coef", key=abs, ascending=False).head(15)
        y_label = "Coefficient"
        title = "Top 15 Features by |Coefficient| (RidgeCV)"
        
    elif model_type == 'GradientBoosting':
        # GradientBoosting uses feature_importances_
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coef": model.feature_importances_
        }).sort_values("coef", ascending=False).head(15)
        y_label = "Feature Importance"
        title = "Top 15 Features by Importance (GradientBoosting)"
        
    elif model_type == 'Random Forest':
        # Random Forest uses feature_importances_
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coef": model.feature_importances_
        }).sort_values("coef", ascending=False).head(15)
        y_label = "Feature Importance"
        title = "Top 15 Features by Importance (Random Forest)"
        
    elif model_type == 'XGBoost':
        # XGBoost uses feature_importances_
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coef": model.feature_importances_
        }).sort_values("coef", ascending=False).head(15)
        y_label = "Feature Importance"
        title = "Top 15 Features by Importance (XGBoost)"
    else:
        st.warning("Feature importance not available for this model type.")
        coef_df = None
    
    if coef_df is not None:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.barh(coef_df["feature"], coef_df["coef"])
        ax3.set_xlabel(y_label, fontsize=12)
        ax3.set_title(title, fontsize=14, fontweight='bold')
        ax3.invert_yaxis()  # so largest is at the top
        ax3.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig3)
        
        # Store top features for graph context
        top_features_list = coef_df["feature"].tolist()
        top_features_values = coef_df["coef"].tolist()
        
    st.write("For linear models, the bars represent the size of each feature's coefficient. "
            "For tree-based models, the bars show feature importance scores based on how much each feature improves the model's accuracy during training. "
            "Larger bars mean the feature contributes more to the prediction.")
    
    # Store graph context in session state for AI chatbot
    predictions = model.predict(X)
    residuals = y - predictions
    st.session_state['graph_context'] = {
        'actual_vs_predicted': {
            'correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'y_range': [float(y.min()), float(y.max())],
            'predictions_range': [float(predictions.min()), float(predictions.max())]
        },
        'residuals': {
            'mean': mean_residual,
            'std': std_residual,
            'min': float(residuals.min()),
            'max': float(residuals.max()),
            'skewness': float(residual_skewness),
            'kurtosis': float(residual_kurtosis)
        },
        'feature_importance': {
            'top_features': top_features_list,
            'top_values': [float(v) for v in top_features_values],
            'chart_type': y_label if y_label else 'Not available'
        }
    }
    st.session_state['graphs_generated'] = True

# Graph button - outside the trigger_2 block so it's always visible
if 'model' in st.session_state and 'X' in st.session_state and 'y' in st.session_state:
    trigger_3 = st.button("Graph", key="button_3")
    if trigger_3:
        display_graphs()
    
    # Display graphs if they were previously generated (persist across reruns)
    # Only show if button wasn't just clicked (to avoid double display)
    elif st.session_state.get('graphs_generated', False):
        display_graphs()


# Initialize OpenAI client (only if API key is available)
client = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if api_key:
        client = OpenAI(api_key=api_key)
except Exception as e:
    client = None
    # Silently fail - user can still use the app without OpenAI

# Build model summary from session state if model exists
if 'model' in st.session_state and 'X' in st.session_state and 'y' in st.session_state:
    model = st.session_state['model']
    X = st.session_state['X']
    y = st.session_state['y']
    model_type = st.session_state.get('model_type', 'Unknown')
    
    r2_score = model.score(X, y)
    
    # Get feature importance/coefficients based on model type
    if model_type in ['LassoCV', 'RidgeCV']:
        feature_info = f"Top 5 coefficients: {sorted(zip(X.columns, model.coef_), key=lambda x: abs(x[1]), reverse=True)[:5]}"
    elif model_type == 'ElasticNetCV':
        enet = model.named_steps['elasticnetcv']
        feature_info = f"Top 5 coefficients: {sorted(zip(X.columns, enet.coef_), key=lambda x: abs(x[1]), reverse=True)[:5]}"
    elif model_type in ['GradientBoosting', 'Random Forest', 'XGBoost']:
        feature_info = f"Top 5 features: {sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]}"
    else:
        feature_info = "Feature information not available"
    
    model_summary = f"""
Models trained: LassoCV, ElasticNetCV, RidgeCV, GradientBoosting, Random Forest, XGBoost
Selected model: {model_type}
R2 Score: {r2_score:.3f}
{feature_info}
"""
    
    # Add graph context if available
    if 'graph_context' in st.session_state:
        graph_ctx = st.session_state['graph_context']
        actual_pred = graph_ctx['actual_vs_predicted']
        residuals = graph_ctx['residuals']
        features = graph_ctx['feature_importance']
        
        graph_summary = f"""

GRAPH ANALYSIS:
1. Actual vs Predicted Plot:
   - Correlation between actual and predicted: {actual_pred['correlation']:.3f}
   - Mean Absolute Error (MAE): {actual_pred['mae']:.3f}
   - Root Mean Squared Error (RMSE): {actual_pred['rmse']:.3f}
   - Actual values range: {actual_pred['y_range'][0]:.2f} to {actual_pred['y_range'][1]:.2f}
   - Predicted values range: {actual_pred['predictions_range'][0]:.2f} to {actual_pred['predictions_range'][1]:.2f}

2. Residuals Plot:
   - Mean residual: {residuals['mean']:.3f} (should be close to 0)
   - Standard deviation of residuals: {residuals['std']:.3f}
   - Residual range: {residuals['min']:.3f} to {residuals['max']:.3f}
   - Skewness: {residuals['skewness']:.3f} (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
   - Kurtosis: {residuals['kurtosis']:.3f} (0 = normal distribution, >0 = heavy tails, <0 = light tails)

3. Feature Importance/Coefficients Chart:
   - Chart type: {features['chart_type']}
   - Top features: {', '.join(features['top_features'][:10])}
   - Top feature values: {[f'{f}: {v:.4f}' for f, v in zip(features['top_features'][:5], features['top_values'][:5])]}
"""
        model_summary += graph_summary
else:
    model_summary = "No model has been trained yet. Please train a model first."
st.subheader("🔎 Ask the AI to explain the results")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Ask something about these models or metrics...")
if user_msg:
    #st.session_state.chat_history.append({"role": "user", "content": user_msg})

    # Call OpenAI with your results as context
    if client is None:
        st.error("OpenAI API key not configured. Please set OPENAI_API_KEY in Streamlit secrets.")
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-5.1",  # or "gpt-3.5-turbo" for cheaper option
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an ML assistant inside a Streamlit scouting dashboard. "
                            "Explain model results simply, mention metrics, and keep answers short."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Here is a summary of the model results:\n{model_summary}\n\n"
                            f"The user asks: {user_msg}"
                        ),
                    },
                ],
            )

            ai_reply = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

            with st.chat_message("assistant"):
                st.markdown(ai_reply)
        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")