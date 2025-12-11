from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error



# ------------------------------------------------------------------------------
# 1. THEME & STYLING (Keep this)
# ------------------------------------------------------------------------------
theme = {
    'background': '#111111',    # Very dark grey
    'card_bg': '#1a1a1a',       # Slightly lighter for cards
    'text': '#FFFFFF',          # White text
    'accent': '#D4AF37',        # Metallic Gold
    'accent_secondary': '#C5A028',
    'font_family': 'Helvetica, Arial, sans-serif'
}

# Styles for Tabs to make them Dark/Gold
tab_style = {
    'borderBottom': f'1px solid {theme["accent"]}',
    'padding': '6px',
    'backgroundColor': theme['card_bg'],
    'color': theme['text']
}

tab_selected_style = {
    'borderTop': f'1px solid {theme["accent"]}',
    'borderBottom': '1px solid #111111',
    'color': '#000000', 
    'fontWeight': 'bold',
    'padding': '6px'
}

# ------------------------------------------------------------------------------
# 2. DATA LOADING (Old data removed)
# ------------------------------------------------------------------------------
# Place your new data loading code here
df = pd.read_csv('final_data_with_demographics.csv')
# Define your two feature sets
feature_sets = {
    'Size': ['face_height', 'face_width', 'nose_width', 'mouth_width'],
    'Ratios': ['face_ratio', 'mouth_nose_ratio', 'eye_ratio', 'golden_score']
}

# jill df
headshots = df.copy() 
# --------------------------------------------------------
# stephanie's code
# --------------------------------------------------------
predictive_cols = ['face_height', 'face_width', 'nose_width', 'mouth_width']
X = df[predictive_cols].copy()
y = df['golden_score'].copy()

# Train/Test split
Xlin_train, Xlin_test, ylin_train, ylin_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))
])
ridge_pipe.fit(Xlin_train, ylin_train)

#Test metrics
ylin_pred_test = ridge_pipe.predict(Xlin_test)
R2_TEST= float(r2_score(ylin_test, ylin_pred_test))
RMSE_TEST = float(np.sqrt(mean_squared_error(ylin_test, ylin_pred_test)))

#Coefficients
coef = ridge_pipe.named_steps['ridge'].coef_
intercept= float(ridge_pipe.named_steps['ridge'].intercept_)
coef_map= dict(zip(predictive_cols, coef))
alpha_selected= float(ridge_pipe.named_steps['ridge'].alpha_)

# artifacts for callbacks
RIDGE_ARTIFACTS = {
    'Xlin_train': Xlin_train,
    'Xlin_test': Xlin_test,
    'ylin_train': ylin_train,
    'ylin_test': ylin_test,
    'ridge_pipe':ridge_pipe,
    'predictive_cols': predictive_cols,
    'ylin_pred_test': ylin_pred_test,
    'R2_TEST': float(R2_TEST),
    'RMSE_TEST': RMSE_TEST,
    'coef_map': coef_map,
    'intercept': intercept,
    'alpha_selected': alpha_selected,
}
# --------------------------------------------------------
# SHEYI's CODE
# --------------------------------------------------------
faces_cut= pd.read_csv("final_data_with_demographics.csv")

# cut golden score into 3 catagories 
faces_cut["golden_score"]  = pd.qcut(faces_cut["golden_score"], q=3, labels=False)

# predictors and predictees
X_knn = faces_cut[["face_ratio", "race", "mouth_nose_ratio","eye_ratio", "gender" ]]
y_knn = faces_cut["golden_score"]

#train and test
Xknn_train, Xknn_test, yknn_train, yknn_test = train_test_split(
    X_knn, y_knn, test_size=0.25, random_state=42, stratify=y_knn
)

# numeric 
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# catagorical
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# preprocessor 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_transformer, make_column_selector(dtype_include=object))
    ]
)

# initial pipes
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(weights="distance"))
])

# fit model 
param_grid = {
    "knn__n_neighbors": range(1, 20, 2),
    "preprocessor__num__imputer__strategy": ["mean", "median"]
}
grid = GridSearchCV(pipe, param_grid, cv=4, scoring="balanced_accuracy", n_jobs=-1,error_score='raise')
grid.fit(Xknn_train, yknn_train)

# find best k 
results_df = pd.DataFrame(grid.cv_results_)

results_df["k"] = results_df["param_knn__n_neighbors"]
results_df["mean_score"] = results_df["mean_test_score"]

best_k = grid.best_params_["knn__n_neighbors"]
best_score = grid.best_score_

# --- GOLD STYLE KNN LINE CHART ---
bestk_fig = px.line(
    results_df,
    x="k",
    y="mean_score",
    title=f"Cross-Validated Balanced Accuracy vs. K (best k = {best_k})",
    markers=True,
    labels={"k": "Number of Neighbors (k)", "mean_score": "Mean CV Balanced Accuracy"},
    template='plotly_dark'
)

# Update line color to Gold
bestk_fig.update_traces(line_color=theme['accent'], marker_color=theme['accent'])

bestk_fig.add_scatter(
    x=[best_k],
    y=[best_score],
    mode="markers+text",
    text=[f"Best k = {best_k}"],
    textposition="top center",
    name="Best k",
    marker=dict(color='white', size=12, symbol='star') # distinct marker for best K
)

bestk_fig.update_layout(
    plot_bgcolor=theme['card_bg'],
    paper_bgcolor=theme['background'],
    font_color=theme['text']
)

#fit with best k 
pipe2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(
        n_neighbors=best_k,
        weights="distance")
        )
])

pipe2.fit(Xknn_train, yknn_train)
yknn_pred = pipe2.predict(Xknn_test)

knn_acc = accuracy_score(yknn_test, yknn_pred)
knn_bal_acc = balanced_accuracy_score(yknn_test, yknn_pred)

df_cat= faces_cut.copy()
#catagory for discrete colors 
df_cat['golden_score']= df_cat['golden_score'].astype("category")

# --- GOLD PALETTE FOR SCATTER PLOTS ---
# Gold, White, Dark Gold
gold_discrete_sequence = [theme['accent'], '#FFFFFF', '#FCF6BA']

eyeVsMouth = px.scatter(
    df_cat, x="eye_ratio", y="mouth_nose_ratio",
    title= "Eye Ratio vs Mouth Nose Ratio",
    color="golden_score",
    color_discrete_sequence=gold_discrete_sequence, 
    symbol="golden_score",
    template='plotly_dark'
)
eyeVsMouth.update_layout(plot_bgcolor=theme['card_bg'], paper_bgcolor=theme['background'], font_color=theme['text'])

faceVsEye = px.scatter(
    df_cat, x="face_ratio", y="eye_ratio", 
    title="Face Ratio vs Eye Ratio",
    color="golden_score",
    color_discrete_sequence=gold_discrete_sequence, 
    symbol="golden_score",
    template='plotly_dark'
)
faceVsEye.update_layout(plot_bgcolor=theme['card_bg'], paper_bgcolor=theme['background'], font_color=theme['text'])

faceVsMouth = px.scatter(
    df_cat, x="face_ratio", y="mouth_nose_ratio",
    title="Face Ratio vs Mouth Nose Ratio", 
    color="golden_score",
    color_discrete_sequence=gold_discrete_sequence, 
    symbol="golden_score",
    template='plotly_dark'
)
faceVsMouth.update_layout(plot_bgcolor=theme['card_bg'], paper_bgcolor=theme['background'], font_color=theme['text'])
# ---------------------------------------------------------
# 2. HELPER FUNCTION: RUN PCA
# ---------------------------------------------------------
def calculate_pca(feature_list):
    """
    Takes a list of columns, runs PCA, and returns the Scores and Loadings.
    """
    # 1. Select and Standardize
    X = df[feature_list].dropna().astype(float)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # 2. Covariance & Eigenvectors
    cov_matrix = np.cov(X_std.T)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    
    # 3. Sort Eigenvalues (High to Low)
    sorted_indices = np.argsort(eigvals)[::-1]
    sorted_vals = eigvals[sorted_indices]
    sorted_vecs = eigvecs[:, sorted_indices]
    
    # 4. Project Data (Get Scores)
    scores = X_std @ sorted_vecs
    
    # 5. Calculate Loadings (Correlation with original variables)
    # Loadings = Eigenvector * sqrt(Eigenvalue)
    loadings = sorted_vecs * np.sqrt(sorted_vals)
    
    # Create temporary DataFrames
    pc_df = df.loc[X.index].copy()
    pc_df['PC1'] = scores[:, 0]
    pc_df['PC2'] = scores[:, 1]
    
    loadings_df = pd.DataFrame(loadings[:, :2], index=feature_list, columns=['PC1', 'PC2'])
    
    return pc_df, loadings_df, sorted_vals

# ------------------------------------------------------------------------------
# 3. APP INIT
# ------------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# ------------------------------------------------------------------------------
# 4. APP LAYOUT
# ------------------------------------------------------------------------------
app.layout = html.Div(
    style={'backgroundColor': theme['background'], 'minHeight': '100vh', 'padding': '20px'},
    children=[
        dbc.Container([
            
            # TITLE (Notice we use className here, but no html.Style block above it)
            html.H1(
                "Hot or Not?!", 
                className="gold-shimmer",  # <--- This class now works because of the css file
                style={'textAlign': 'center', 'marginBottom': '30px'} 
            ),
            
            # TABS
            dcc.Tabs(id='tabs-on-top', value='tab-7', children=[
                
                # TAB 1
                dcc.Tab(label='About the Data', value='tab-1', style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.H3("About The Data", style={'color': theme['accent']}),
                        dcc.Markdown("""
                            ***What is the Golden Ratio?***
                                     
                                     
                            Aesthetics and appearance play a major role in our society today. While there is no true definition of beauty, our ability to find reliable patterns and proportions that extend into art, aesthetics, and ideals of beauty are seen everywhere today. The golden ratio is one of many theories of what makes an object perceived as pleasing. 

                                          
                            The golden ratio is rooted in a mathematical proportion, the Greek letter φ (phi), which is viewed as the ideal proportion. The proportion mathematically works out that one proportion is related to the other by 1.618. This number is derived from the Fibonacci sequence and other instances in nature. 
                                     
                            ***Our Study***
                                     

                            In the nature of facial aesthetics, the golden ratio is one of many frameworks for perceived attractiveness through balance and harmony in features. Although there are many different ratios that can be viewed and analyzed, in our study ,we focused on measurements between face length to width, eye spacing, and nose to mouth ratio. This is only a subset of the proportions deemed as “ideal”. 
         

                           ***Our Data***
                                     

                            We created our own dataset using data from our cohort. We collected information such as name, email, race, gender, and a headshot photo from 52 individuals within our cohort. We then used a convolutional neural network including python packages from opencv and mediapipe to build a complete dataset featuring measurements such as mouth width, nose width, face width, and face height. These features were then used to create face and mouth-nose ratios and averaged to create an overall “golden ratio score” or proportionality score. This complete dataframe is used for all of our models. 

                            ***Complications and Caveats***
                                     

                            Because of inconsistency in headshot files, posture, and quality of images, not all measurements from all individuals were able to be taken. In addition, small changes in posture and quality could affect our CNNs ability to correctly detect facial features. facial feature detection trial and error with using different networks was frustrating to say the least. OpenCV facial width and height originated as a square, eyebrows would be recognized as noses, background noise from images would be detected as faces themselves, or nothing would be detected at all. We went through over three different CNNs before settling on one and being satisfied that most faces were detected accurately. Moreover, we had a very small sample size, which most likely indicates that our analysis is not an accurate representation of the cohort. 

              
                            """, style={'color': theme['text']}),
                        
                        # Github Link
                        html.A(
                            DashIconify(icon="ion:logo-github", width=30, color=theme['accent']),
                            href="https://github.com/sek2dcs/DS6021-ML-Final-Project", # Update with your new link
                            target="_blank"
                        )
                    ], style={'padding': '20px'})
                ]),
                
                # TAB 2
                dcc.Tab(label='Dataset', value='tab-2', style=tab_style, selected_style=tab_selected_style, children=[
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.H3("Cohort Data", style={'color': theme['accent']}),
                            dash_table.DataTable(
                                id='dataset-table',
                                data=df.to_dict('records'),
                                columns=[{"name": i, "id": i} for i in df.columns],
                                page_size=10,
                                style_table={'overflowX': 'auto'},
                                style_header={
                                    'backgroundColor': theme['card_bg'],
                                    'color': theme['accent'],
                                    'fontWeight': 'bold',
                                    'border': '1px solid #333'
                                },
                                style_data={
                                    'backgroundColor': theme['background'],
                                    'color': theme['text'],
                                    'border': '1px solid #333'
                                },
                            )
                        ], width=12),
                    ])
                ]),
                
                # TAB 3
                dcc.Tab(label='Linear Regression', value='tab-3', style=tab_style, selected_style=tab_selected_style, children=[
                    html.Br(),
                    dbc.Container([
                    html.H1(
                        "Linear Regression",
                        style={'textAlign': 'center', 'marginBottom': '16px', 'color': theme['accent']}
                    ),
                    html.H3("Research Question:", style={'color': theme['accent'], 'textAlign': 'center', 'marginBottom': '6px'}),
                    html.P(
                    "Do raw facial dimensions contain enough information to find golden ratio without using ratio based features?",
                    style={'color': theme['text'], 'textAlign': 'center', 'fontSize': '18px', 'marginBottom': '24px'}
                    ),
                    dbc.Row([
                    dbc.Col([
                        html.Label("Select a predictor (for visualization):", style={'color': theme['text']}),
                        dcc.Dropdown(
                            id='regression-variable',
                            options=[{'label': col, 'value': col} for col in predictive_cols],
                            value=predictive_cols[0],
                            style={'color': '#000'}
                        )
                        ], width=6),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([dcc.Graph(id='lm-scatter',style={'height': '500px'})], width=6),
                        dbc.Col([dcc.Graph(id='lm-residuals',style={'height': '500px'})], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                id='model-summary',
                                style={
                                    'padding': '16px',
                                    'backgroundColor': theme['card_bg'],
                                    'border': f'1px solid {theme["accent"]}',
                                    'borderRadius': '8px',
                                    'marginTop': '20px'
                                }
                            )
                         ], width=12)
                    ]),
                    ], fluid=True)
    ]
), 
                
                # TAB 4
                dcc.Tab(label='Logistic Regression', value='tab-4', style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.H3("Logistic Regression Model", style={'color': theme['accent']}),
                        html.P("Predicting golden ratio categories based on facial features", style={'color': theme['text']}),
                        html.Br(),
                        
                        # Number of Bins Slider
                        html.H4("Select Number of Bins", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Slider(
                            id='num-bins-slider',
                            min=3,
                            max=5,
                            step=1,
                            value=5,
                            marks={i: str(i) for i in range(3, 6)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        html.Br(),
                        
                        # Check Individual's Golden Ratio Bin
                        html.H4("Check Individual's Golden Ratio Bin", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Dropdown(
                            id='person-dropdown',
                            options=[{'label': name, 'value': name} for name in sorted(df['name'].unique())],
                            placeholder="Select a person",
                            style={
                                'backgroundColor': '#ffff',  
                                'color': '#000000',                 
                                'border': '1px solid #333'           
                            }
                        ),
                        html.Div(id='person-bin-output', style={'color': theme['text'], 'marginTop': '10px', 'fontSize': '16px'}),
                        html.Br(),
                        html.Hr(style={'borderColor': theme['accent']}),
                        html.Br(),
                        
                        # Model Performance Metrics
                        html.H4("Model Performance", style={'color': theme['accent'], 'marginTop': '20px'}),
                        html.Div(id='logistic-metrics', style={'color': theme['text'], 'marginBottom': '20px'}),
                        html.Br(),
                        
                        # Confusion Matrix
                        html.H4("Confusion Matrix", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Graph(id='logistic-confusion-matrix'),
                        html.Br(),
                        
                        # ROC Curves
                        html.H4("ROC Curves", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Graph(id='logistic-roc-curves'),
                        html.Br(),
                        
                        # Feature Importance
                        html.H4("Feature Coefficients by Category", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Graph(id='logistic-feature-importance'),
                        html.Br(),
                        
                        # Ranked Feature Importance
                        html.H4("Overall Feature Importance", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Graph(id='logistic-feature-ranked'),
                        html.Br(),
                        
                        # Model Coefficients Table
                        html.H4("Model Coefficients & Intercepts", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dash_table.DataTable(
                            id='logistic-coefficients-table',
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': theme['card_bg'],
                                'color': theme['accent'],
                                'fontWeight': 'bold',
                                'border': '1px solid #333'
                            },
                            style_data={
                                'backgroundColor': theme['background'],
                                'color': theme['text'],
                                'border': '1px solid #333'
                            },
                        ),
                        html.Br(),
                        
                        # Conclusions
                        html.H4("Conclusions", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Markdown("""
### Logistic Regression Interpretation

This logistic regression model answers the question, "How accurately can a logistic regression model classify individuals in our cohort into golden ratio categories based on their features, and how does the number of bins affect model performance?"
We can see that mouth_nose ratio and face_ratio are the most important coefficients to determine what bin an individual goes in. In the feature importance model, we can see that the bins that lie on the extremities usually have the greatest coefficients.
While looking at our confusion matrix, this remains true. Our model does the the best when classifying individuals into the bins that indicate the extremities (very close or very far) but in the moderate category it does not do as well. We can see a similar output in the ROC curves and AUC values. The model does the best when classifying individuals into the extreme bins which is denoted by the ROC curves that are closest to the top left corner and AUC values that are higher and closer to 1.
Because our sample size is so small, creating bins was difficult as we needed enough data to be included in the training set and in the test set. That is why our ROC curve looks more similar to a stepwise function - because there is such a limited sample the ROC curve is plotting discrete values.
                        """, style={'color': theme['text'], 'padding': '15px', 'backgroundColor': theme['card_bg'], 'borderRadius': '6px'})
                    ], style={'padding': '20px'})
                ]),

                # TAB 5 
                dcc.Tab(label='KNN', value='tab-5', style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.H3("Question:", style={'color': theme['accent']}),
                        html.P("Can the KNN model accurately and consistently group subjects into the right golden score group?", style={'color': theme['text']}),
                        dcc.Graph(
                            figure=bestk_fig,
                            style={'width': '80%', 'height': 'auto'},
                            responsive= True

                        ),
                        dcc.Slider(id='bestK_slider', min=2, max=10, step=1, value=5, marks={i: str(i) for i in range(2, 11)}, tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Div(id='slider-output-container'),

                        #space
                        html.Div(style={'height': '20px'}),

                        html.Img(src='assets/knn_confusionMatrix.png', style={'width': '60%', 'height': 'auto'}),

                    
                        
                            dbc.Col(
                                dcc.Graph(
                                    figure= eyeVsMouth,
                                    style={'width': 'auto', 'height': '60vh'}
                                )
                            ),
                            dbc.Col(
                                dcc.Graph(figure=faceVsEye,style={'width': 'auto', 'height': '60vh'} )
                            ),
                            dbc.Col(
                                dcc.Graph(figure= faceVsMouth ,style={'width': 'auto', 'height': '60vh'})
                        
                            ),
                        
                        

                        html.H3("Conclusion:", style={'color': theme['accent']}),
                        html.P("I would say that the KNN model does a pretty good job at correctly clustering items into the correct group since they accuracy is 0.846 and the balanced accuracy is 0.850. The model benefitted from splitting the golden score into 3 buckets instead of 5. I would say there is still some room for improvement.", style={'color': theme['text']}),

                        # Github Link
                        html.A(
                            DashIconify(icon="ion:logo-github", width=30, color=theme['accent']),
                            href="#", # Update with your new link
                            target="_blank"
                        )
                    ], style={'padding': '20px'})
                ]), 
                # TAB 6
                dcc.Tab(label='K means', value='tab-6', style=tab_style, selected_style=tab_selected_style, children=[
                    html.Div([
                        html.Br(), 
                        html.H2("Are there similarities in face shape between Gender and Sex ?", style={'color': theme['accent']}), 
                        dcc.Markdown("""
                                    Using K means clustering to see if there are natural patterns in the data. Seeing if clusters are affected by data demographics. 

                                    
                                     Please note that elbow/ silhoutte plots state that best fit for this dataset is to use 2 clusters. Included slider to make data more interactive.
                                     """), 
                        html.Br(),
                        
                        # K Value Slider
                        html.H4("Select Number of Clusters (K)", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Slider(
                            id='k-value-slider',
                            min=2,max=8, step=1,value=2,
                            marks={i: str(i) for i in range(2, 9)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        html.Br(),
                        
                        # Clusters Visualized
                        html.H4("K-Means Clusters Visualization", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Graph(id='cluster-scatter-plot'),
                        html.Br(),
                        
                        # Cluster Gender
                        html.H4("Cluster Distribution by Gender", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Graph(id='cluster-gender-plot'),
                        html.Br(),
                        
                        # Cluster Race
                        html.H4("Cluster Distribution by Race", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Graph(id='cluster-race-plot'),
                        html.Br(),

                        
                        # Conclusions
                        html.H4("Conclusions", style={'color': theme['accent'], 'marginTop': '20px'}),
                        dcc.Markdown("""
                                    ### K- means model Interpretation

                                    K-means clustering is a unsupervised learning technique used to find natural patterns within the data. Our data favors using 2-3 as the number of clusters as per our silhouettte and elbow plot (not included on dashboard). We facial proportions such as face length-width, nose-mouth, and eye ditance ratios as numeric input values. 
                                    Once clusters were fit we tried to see if there was correlation between the output clusters and the demographics of our sample. 
                                    While using 2 clusters, you can see patterns of specific genders or races being correlated within different cluster. 
                                    This leads us to believe that different races and gender are predisposed to different facial features or proportions. 
                                    In terms of the golden ratio, this could imply that the ratio is biased and therefore an irrealistic standard in our society today 
                                                                  
                        """, style={'color': theme['text'], 'padding': '15px', 'backgroundColor': theme['card_bg'], 'borderRadius': '6px'})
                ])]), 
                # TAB 7 
                dcc.Tab(label='PCA', value='tab-7', style=tab_style, selected_style=tab_selected_style, children=[
                    html.Br(),
                    html.Div([
                        html.Label("Select Analysis Model:", style={'color': theme['text']}),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': 'Size & Dimensions', 'value': 'Size'},
                                {'label': 'Ratios & Golden Score', 'value': 'Ratios'}
                            ],
                            value='Size', 
                            clearable=False,
                            style={'color': '#000'} # Text inside dropdown needs to be black
                        )
                    ], style={'width': '50%', 'margin': 'auto', 'paddingBottom': '20px'}),
    
                    # PCA Loadings Table
                    html.H4("PCA Loadings", style={'color': theme['accent'], 'textAlign': 'center'}),
                    dash_table.DataTable(
                        id='loadings-df',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'marginBottom': '30px'},
                        # Add Dark Theme Styling
                        style_header={'backgroundColor': theme['card_bg'], 'color': theme['accent'], 'border': '1px solid #333'},
                        style_data={'backgroundColor': theme['background'], 'color': theme['text'], 'border': '1px solid #333'},
                    ),
                    dcc.Graph(id='pca-graph', style={'height': '70vh'}), 
                    html.Br(), 
                    html.H4("PCA Conclusions", style={'color': theme['accent'], 'textAlgin': 'center'}), 
                    dcc.Markdown("""
                            ***Size & Dimensions PCA***
                                 

                            The Size & Dimensions PCA model answers the question of how proportional the individual's face is. Based on the biplot, PC1 describes the 'largeness' of one's face. The left side of the x axis indicates people with smaller faces (aka small face height, small face width, small nose width and small mouth width), and the right side of the x axis indicates the opposite. The PC2 in this case mostly showcases the proportion of mouth width to face size. In other words, the top of the y axis indicates a larger face with a relatively narrow mouth, and the bottom of the y axis indicates a smaller face with a relatively wider mouth. The closer the point is to (0,0), the more proportional the mouth width is to face size. 

                                 
                            ***Ratios & Golden Score PCA***
                                 

                            The Ratio & Golden Score PCA model answers the question of which face ratio and feature ratios are most ideal to fit the golden ratio. Based on the biplot, PC1 describes the relationship of the golden score based on the mouth vs nose ratio. The right side of the x axis indicates wider features, namely a wider mouth than nose. Because the golden score line is also pointing to the right, the right side of the x axis also indicates a higher golden score. The left side of the x axis indicates a wider nose than mouth and a lower golden score. PC2 indicates the face length, with the top of the y axis being longer faces and the bottom of the y axis being shorter faces. There is also the contribution of the eye_ratio variable (which didn't contribute to the golden score), which also shows in the PC2 plot as shorter faces typically having wider set eyes.
                            
                                 
                            The top right of the biplot indicates faces that are the closest to the golden ratio. These faces are typically longer (based on PC2) and more balanced nose/mouth ratio (though leaning more towards a wider mouth versus nose). The top left of the biplot indicates faces that are longer, but have more narrow features, particularly a wider nose than mouth. The bottom right of the biplot indicates shorter faces with wider features (particuarly wider eyes and wider mouth). The bottom right has particuarly good golden scores, but not as good as the top right. The bottom left of the biplot indicates faces that are shorter and have more narrow set features, meaning they have the smallest golden scores. 
                            In summary, the golden score is the highest for longer faces that have wider features, particularly a wider nose than mouth. 
                    
                            """, style={'color': theme['text']})
                ])
            ])
        ], fluid=True)
    ]
)

# ------------------------------------------------------------------------------
# 5. CALLBACKS (Old callbacks removed)


#Jills code

def k_means_func(num_clust):

    X = headshots[['face_ratio', 'eye_ratio', 'mouth_nose_ratio']].copy()
    X = X.dropna()
    
    # Get the indices of rows that were used (non-NaN)
    valid_indices = X.index

    # kmeans
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=num_clust, init="k-means++", n_init=10, random_state=123))
    ])
    pipe.fit(X)
    labels = pipe["kmeans"].labels_

    # error handing bc i broke it when i put it in dashbaord :/
    headshots_clustered = headshots.copy()
    if "cluster" in headshots_clustered.columns:
        headshots_clustered = headshots_clustered.drop(columns=["cluster"])
    headshots_clustered["cluster"] = None
    headshots_clustered.loc[valid_indices, "cluster"] = labels.astype(str)
    
    headshot3 = headshots_clustered[headshots_clustered["cluster"].notna()][["name", 'race', 'gender', 'cluster', "golden_score"]].copy()
    
    headshot3['cluster'] = headshot3['cluster'].astype(str)
    headshot3['gender'] = headshot3['gender'].astype(str).str.strip()
    
    headshot3 = headshot3.drop_duplicates(subset=['name', 'cluster'], keep='first')
    
    # Additional check: ensure each person appears only once (take their assigned cluster)
    if headshot3['name'].duplicated().any():
        headshot3 = headshot3.drop_duplicates(subset=['name'], keep='first')

    # Clusters visualized
    headshots_plot = headshots_clustered[headshots_clustered["cluster"].notna()].copy()
    fig_clusters = px.scatter(
        headshots_plot,
        x="face_ratio",
        y="mouth_nose_ratio",
        hover_name="name",
        hover_data=["name", 'race', 'gender', 'cluster', "golden_score"],
        height=650,
        color='cluster',
        title="K-Means Clusters Across Cohort",
        template='plotly_dark'
    )
    fig_clusters.update_layout(
        plot_bgcolor=theme['card_bg'],
        paper_bgcolor=theme['background'],
        font_color=theme['text'])

    crosstab = pd.crosstab(headshot3['gender'], headshot3['cluster'], normalize='columns', dropna=False)
    
    # AI added bc i broke it
    cluster_columns = [str(i) for i in range(num_clust)]
    for col in cluster_columns:
        if col not in crosstab.columns:
            crosstab[col] = 0.0
    
    crosstab = crosstab[cluster_columns]
    crosstab = round(crosstab, 2).reset_index()
    cluster_gender = pd.melt(crosstab, id_vars='gender', value_vars=cluster_columns, 
                            var_name='cluster', value_name='value')
    cluster_gender = cluster_gender.drop_duplicates(subset=['gender', 'cluster'])
    cluster_gender = cluster_gender.sort_values(['cluster', 'gender'])
    cluster_gender['percentage'] = (cluster_gender['value'] * 100).round(1)
    cluster_gender['bar_text'] = cluster_gender['percentage'].apply(lambda x: str(int(round(x, 0)))) + "%"

    fig_cluster_gender = px.bar(
        cluster_gender, x='cluster', y='value', color='gender',
        text='bar_text',barmode='group',
        title="Cluster Distribution by Gender",
        labels={'value': 'Proportion', 'cluster': 'Cluster'},
        template='plotly_dark'
    )
    fig_cluster_gender.update_layout(
        showlegend=True,
        plot_bgcolor=theme['card_bg'],
        paper_bgcolor=theme['background'],
        font_color=theme['text']
    )

    # Cluster race 
    cluster_race = round(pd.crosstab(headshot3['cluster'], headshot3['race'], normalize='columns'), 2).reset_index()
    cluster_race = pd.melt(cluster_race, id_vars='cluster', 
        value_vars=['asian', 'black or african american', 'hispanic or latino',
                    'middle eastern or north african', 'other/mixed', 'white/caucasian']
    )
    cluster_race['percentage'] = (cluster_race['value'] * 100).round(1)
    cluster_race['bar_text'] = cluster_race['percentage'].apply(lambda x: str(int(round(x, 0)))) + "%"

    fig_cluster_race = px.bar(
        cluster_race,x='cluster',  y='value', color='race',
        text='bar_text', barmode='group', width=900, height=400,
        title="Cluster Distribution by Race",
        labels={'value': 'Proportion', 'cluster': 'Cluster'},
        template='plotly_dark'
    )
    fig_cluster_race.update_layout(
        showlegend=True,
        plot_bgcolor=theme['card_bg'],
        paper_bgcolor=theme['background'],
        font_color=theme['text']
    )
    
    return fig_clusters, fig_cluster_gender, fig_cluster_race
# bella code
BIN_LABELS = {
    3: ['Close', 'Moderate', 'Far'],
    4: ['Very Close', 'Close', 'Moderate', 'Far'],
    5: ['Very Close', 'Close', 'Moderate', 'Far', 'Very Far']
}

def recategorize_y(df, num_bins):
    """Bin golden_score into meaningful class names based on number of bins"""
    labels = BIN_LABELS[num_bins]
    quantiles = [df['golden_score'].quantile(i / num_bins) for i in range(1, num_bins)]
    df['golden_ratio_category'] = pd.cut(
        df['golden_score'],
        bins=[-float('inf')] + quantiles + [float('inf')],
        labels=labels
    )
    return df

# Remove rows with missing face measurements or golden_score
df_clean = df.dropna(subset=['face_width', 'face_height', 'golden_score']).copy()

# Calculate face ratio (height/width) - already exists but recalculating for consistency
df_clean['face_ratio'] = df_clean['face_height'] / df_clean['face_width']

# Standardize pixel measurements by converting to ratios relative to face width
df_clean['nose_to_face_ratio'] = df_clean['nose_width'] / df_clean['face_width']
df_clean['mouth_to_face_ratio'] = df_clean['mouth_width'] / df_clean['face_width']

# Initial categorization with 5 bins (default)
df_clean = recategorize_y(df_clean.copy(), 5)

# Since headshots are different sizes, we'll use ratios instead of raw pixel measurements
# This standardizes the features across different image sizes
# Select ratio-based features (standardized to be size-independent)
ratio_features = ['face_ratio', 'mouth_nose_ratio', 'eye_ratio', 
                  'nose_to_face_ratio', 'mouth_to_face_ratio']

# Use rows where we have at least face_width and face_height
df_model = df_clean.dropna(subset=['face_width', 'face_height']).copy()

# Handle missing values in ratio features by imputing with median
for col in ratio_features:
    if col in df_model.columns and df_model[col].isnull().sum() > 0:
        median_val = df_model[col].median()
        df_model[col] = df_model[col].fillna(median_val)

# Encode categorical variables (gender, race) if they exist
# Check unique values in gender
if 'gender' in df_model.columns:
    # Create dummy variables for gender
    gender_dummies = pd.get_dummies(df_model['gender'], prefix='gender')
    df_model = pd.concat([df_model, gender_dummies], axis=1)
    ratio_features = ratio_features + list(gender_dummies.columns)

# Update feature columns to include all ratio features
feature_columns = [col for col in ratio_features if col in df_model.columns]

# Prepare X and y
X = df_model[feature_columns]
y = df_model['golden_ratio_category']

# Split the data
# Note: Not using stratify because some categories may have very few samples
# which would cause an error with stratified splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features (important for logistic regression)
# Even though we're using ratios, StandardScaler ensures all features are on the same scale
# This is especially important when features have different ranges
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
# Using multinomial for multi-class classification
logistic_model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = logistic_model.predict(X_train_scaled)
y_test_pred = logistic_model.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(
    cm,
    index=logistic_model.classes_,
    columns=logistic_model.classes_
)

confusion_matrix_graph = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale='Blues',
    labels=dict(x="Predicted Label", y="True Label", color="Count"),
    title="Confusion Matrix - Test Set"
)
confusion_matrix_graph.update_layout(
    plot_bgcolor=theme['card_bg'],
    paper_bgcolor=theme['background'],
    font_color=theme['text']
)

# Get predicted probabilities for test set
y_test_proba = logistic_model.predict_proba(X_test_scaled)

# Binarize the labels for multiclass ROC (one-vs-rest approach)
y_test_binarized = label_binarize(y_test, classes=logistic_model.classes_)
n_classes = len(logistic_model.classes_)

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_test_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate micro-averaged ROC curve and AUC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_test_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Calculate macro-averaged AUC
roc_auc["macro"] = roc_auc_score(y_test_binarized, y_test_proba, average='macro', multi_class='ovr')

# Create figure
roc_curve_graph = go.Figure()

# Colors for the classes
colors = ['blue', 'red', 'green', 'orange', 'purple']

# Add ROC curves for each class
for i, color in zip(range(n_classes), colors):
    roc_curve_graph.add_trace(
        go.Scatter(
            x=fpr[i],
            y=tpr[i],
            mode='lines',
            line=dict(color=color, width=2),
            name=f"ROC curve for {logistic_model.classes_[i]} (AUC = {roc_auc[i]:.3f})"
        )
    )

# Add micro-average ROC curve
roc_curve_graph.add_trace(
    go.Scatter(
        x=fpr["micro"],
        y=tpr["micro"],
        mode='lines',
        line=dict(color='deeppink', width=2, dash='dash'),
        name=f"Micro-averaged ROC (AUC = {roc_auc['micro']:.3f})"
    )
)

# Add diagonal random classifier line
roc_curve_graph.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        name="Random Classifier (AUC = 0.500)"
    )
)

# Update layout
roc_curve_graph.update_layout(
    title="ROC Curves for Multiclass Logistic Regression Model",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=900,
    height=700,
    plot_bgcolor=theme['card_bg'],
    paper_bgcolor=theme['background'],
    font_color=theme['text'],
    legend=dict(
        bgcolor=theme['card_bg'],
        bordercolor=theme['accent']
    )
)

# Get feature coefficients for each class
coefficients = logistic_model.coef_
feature_names = feature_columns

# Create a DataFrame to visualize feature importance
coef_df = pd.DataFrame(
    coefficients.T,
    index=feature_names,
    columns=logistic_model.classes_
)

coef_long = coef_df.drop(columns=['avg_abs_coef'], errors='ignore').reset_index()
coef_long = coef_long.melt(id_vars='index',
                           var_name='Class',
                           value_name='Coefficient')
coef_long.rename(columns={'index': 'Feature'}, inplace=True)

# Create grouped bar chart
feature_importance_graph = px.bar(
    coef_long,
    x='Feature',
    y='Coefficient',
    color='Class',
    barmode='group',
    title='Feature Coefficients by Golden Ratio Category',
)
feature_importance_graph.update_layout(
    plot_bgcolor=theme['card_bg'],
    paper_bgcolor=theme['background'],
    font_color=theme['text'],
    xaxis_title="Features",
    yaxis_title="Coefficient Value",
    legend_title="Category",
    xaxis=dict(tickangle=45)
)

# Calculate average absolute coefficient for overall feature importance
coef_df['avg_abs_coef'] = coef_df.abs().mean(axis=1)
coef_df_sorted = coef_df.sort_values('avg_abs_coef', ascending=False)

coef_ranked = coef_df_sorted[['avg_abs_coef']].reset_index()

feature_importance_ranked_graph = px.bar(
    coef_ranked,
    x='index',
    y='avg_abs_coef',
    title='Overall Feature Importance (Average Absolute Coefficient)',
    labels={'index': 'Feature', 'avg_abs_coef': 'Importance'},
)
feature_importance_ranked_graph.update_layout(
    plot_bgcolor=theme['card_bg'],
    paper_bgcolor=theme['background'],
    font_color=theme['text'],
    xaxis=dict(tickangle=45)
)

# Create model coefficients table data
coef_table_df = pd.DataFrame(
    logistic_model.coef_,
    index=logistic_model.classes_,
    columns=feature_columns
)
intercept_df = pd.DataFrame(
    logistic_model.intercept_,
    index=logistic_model.classes_,
    columns=["Intercept"]
)
model_details_df = pd.concat([coef_table_df, intercept_df], axis=1)
model_details_df.reset_index(inplace=True)
model_details_df.rename(columns={'index': 'Class'}, inplace=True)

# bella callback
@app.callback(
    Output('logistic-metrics', 'children'),
    Output('logistic-confusion-matrix', 'figure'),
    Output('logistic-roc-curves', 'figure'),
    Output('logistic-feature-importance', 'figure'),
    Output('logistic-feature-ranked', 'figure'),
    Output('logistic-coefficients-table', 'columns'),
    Output('logistic-coefficients-table', 'data'),
    Input('tabs-on-top', 'value'),
    Input('num-bins-slider', 'value')
)
def update_logistic_plots(tab_value, num_bins):
    if tab_value == 'tab-4':
        df_binned = recategorize_y(df_clean.copy(), num_bins)
        target_col = 'golden_ratio_category'
        ratio_features = ['face_ratio', 'mouth_nose_ratio', 'eye_ratio', 'nose_to_face_ratio', 'mouth_to_face_ratio']
        df_model_binned = df_binned.dropna(subset=['face_width', 'face_height']).copy()
        for col in ratio_features:
            if col in df_model_binned.columns and df_model_binned[col].isnull().sum() > 0:
                median_val = df_model_binned[col].median()
                df_model_binned[col] = df_model_binned[col].fillna(median_val)
        
        if 'gender' in df_model_binned.columns:
            gender_dummies = pd.get_dummies(df_model_binned['gender'], prefix='gender')
            df_model_binned = pd.concat([df_model_binned, gender_dummies], axis=1)
            ratio_features_binned = ratio_features + list(gender_dummies.columns)
        else:
            ratio_features_binned = ratio_features
        
        feature_columns_binned = [col for col in ratio_features_binned if col in df_model_binned.columns]
        X_binned = df_model_binned[feature_columns_binned]
        y_binned = df_model_binned[target_col]
        X_train_binned, X_test_binned, y_train_binned, y_test_binned = train_test_split(X_binned, y_binned, test_size=0.2, random_state=42)
        
        scaler_binned = StandardScaler()
        X_train_scaled_binned = scaler_binned.fit_transform(X_train_binned)
        X_test_scaled_binned = scaler_binned.transform(X_test_binned)
        
        model_binned = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        model_binned.fit(X_train_scaled_binned, y_train_binned)
        
        y_train_pred_binned = model_binned.predict(X_train_scaled_binned)
        y_test_pred_binned = model_binned.predict(X_test_scaled_binned)
        train_accuracy_binned = accuracy_score(y_train_binned, y_train_pred_binned)
        test_accuracy_binned = accuracy_score(y_test_binned, y_test_pred_binned)
        
        metrics_text = html.Div([
            html.P(f"Training Accuracy: {train_accuracy_binned:.4f} ({train_accuracy_binned*100:.2f}%)", style={'fontSize': '16px', 'margin': '5px 0'}),
            html.P(f"Test Accuracy: {test_accuracy_binned:.4f} ({test_accuracy_binned*100:.2f}%)", style={'fontSize': '16px', 'margin': '5px 0'}),
        ])
        
        # --- 1. GOLD CONFUSION MATRIX ---
        cm_binned = confusion_matrix(y_test_binned, y_test_pred_binned, labels=model_binned.classes_)
        cm_df_binned = pd.DataFrame(cm_binned, index=model_binned.classes_, columns=model_binned.classes_)
        
        # Custom Black to Gold Color Scale
        gold_scale = [[0, '#000000'], [1, theme['accent']]]
        
        confusion_matrix_graph_binned = px.imshow(
            cm_df_binned, text_auto=True, 
            color_continuous_scale=gold_scale, # Applied Gold Scale
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            title="Confusion Matrix - Test Set",
            template='plotly_dark'
        )
        confusion_matrix_graph_binned.update_layout(plot_bgcolor=theme['card_bg'], paper_bgcolor=theme['background'], font_color=theme['text'])
        
        # --- 2. GOLD ROC CURVES ---
        y_test_proba_binned = model_binned.predict_proba(X_test_scaled_binned)
        y_test_binarized_binned = label_binarize(y_test_binned, classes=model_binned.classes_)
        n_classes_binned = len(model_binned.classes_)
        fpr_binned, tpr_binned, roc_auc_binned = dict(), dict(), dict()
        
        for i in range(n_classes_binned):
            fpr_binned[i], tpr_binned[i], _ = roc_curve(y_test_binarized_binned[:, i], y_test_proba_binned[:, i])
            roc_auc_binned[i] = auc(fpr_binned[i], tpr_binned[i])
        
        fpr_binned["micro"], tpr_binned["micro"], _ = roc_curve(y_test_binarized_binned.ravel(), y_test_proba_binned.ravel())
        roc_auc_binned["micro"] = auc(fpr_binned["micro"], tpr_binned["micro"])
        
        roc_curve_graph_binned = go.Figure()
        
        # Distinct Gold-ish colors for lines
        colors_binned = [theme['accent'], '#FFFFFF', theme['accent_secondary'], '#FFEA00', '#fcf6ba']
        
        for i, color in zip(range(n_classes_binned), colors_binned[:n_classes_binned]):
            roc_curve_graph_binned.add_trace(go.Scatter(
                x=fpr_binned[i], y=tpr_binned[i], mode='lines',
                line=dict(color=color, width=2),
                name=f"ROC curve for {model_binned.classes_[i]} (AUC = {roc_auc_binned[i]:.3f})"
            ))
        
        roc_curve_graph_binned.add_trace(go.Scatter(
            x=fpr_binned["micro"], y=tpr_binned["micro"], mode='lines',
            line=dict(color='white', width=2, dash='dash'), # Changed to White
            name=f"Micro-averaged ROC (AUC = {roc_auc_binned['micro']:.3f})"
        ))
        
        roc_curve_graph_binned.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(color='grey', width=1, dash='dash'),
            name="Random Classifier"
        ))
        
        roc_curve_graph_binned.update_layout(
            title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            width=900, height=700,
            plot_bgcolor=theme['card_bg'], paper_bgcolor=theme['background'], font_color=theme['text'],
            legend=dict(bgcolor=theme['card_bg'], bordercolor=theme['accent'])
        )
        
        # --- 3. GOLD BAR CHARTS ---
        coefficients_binned = model_binned.coef_
        coef_df_binned = pd.DataFrame(coefficients_binned.T, index=feature_columns_binned, columns=model_binned.classes_)
        coef_long_binned = coef_df_binned.drop(columns=['avg_abs_coef'], errors='ignore').reset_index().melt(id_vars='index', var_name='Class', value_name='Coefficient')
        coef_long_binned.rename(columns={'index': 'Feature'}, inplace=True)
        
        feature_importance_graph_binned = px.bar(
            coef_long_binned, x='Feature', y='Coefficient', color='Class',
            barmode='group', title='Feature Coefficients',
            color_discrete_sequence=colors_binned, # Apply Gold Palette
            template='plotly_dark'
        )
        feature_importance_graph_binned.update_layout(
            plot_bgcolor=theme['card_bg'], paper_bgcolor=theme['background'], font_color=theme['text'],
            xaxis=dict(tickangle=45)
        )
        
        # Ranked Importance
        coef_df_binned['avg_abs_coef'] = coef_df_binned.abs().mean(axis=1)
        coef_ranked_binned = coef_df_binned.sort_values('avg_abs_coef', ascending=False)[['avg_abs_coef']].reset_index()
        
        feature_importance_ranked_graph_binned = px.bar(
            coef_ranked_binned, x='index', y='avg_abs_coef',
            title='Overall Feature Importance',
            labels={'index': 'Feature', 'avg_abs_coef': 'Importance'},
            template='plotly_dark'
        )
        # Force bars to be solid Gold
        feature_importance_ranked_graph_binned.update_traces(marker_color=theme['accent']) 
        feature_importance_ranked_graph_binned.update_layout(
            plot_bgcolor=theme['card_bg'], paper_bgcolor=theme['background'], font_color=theme['text'],
            xaxis=dict(tickangle=45)
        )
        
        # Table
        coef_table_df_binned = pd.DataFrame(model_binned.coef_, index=model_binned.classes_, columns=feature_columns_binned)
        intercept_df_binned = pd.DataFrame(model_binned.intercept_, index=model_binned.classes_, columns=["Intercept"])
        model_details_df_binned = pd.concat([coef_table_df_binned, intercept_df_binned], axis=1).reset_index().rename(columns={'index': 'Class'})
        columns = [{"name": i, "id": i} for i in model_details_df_binned.columns]
        data = model_details_df_binned.to_dict('records')
        
        return (metrics_text, confusion_matrix_graph_binned, roc_curve_graph_binned, 
                feature_importance_graph_binned, feature_importance_ranked_graph_binned, columns, data)

    empty_fig = {'data': [], 'layout': {'title': 'Select Logistic Regression tab to view'}}
    return (html.Div(""), empty_fig, empty_fig, empty_fig, empty_fig, [], [])
# bella callback pt 2
@app.callback(
    Output('person-bin-output', 'children'),
    Input('person-dropdown', 'value'),
    Input('num-bins-slider', 'value')
)
def show_person_bin(selected_name, num_bins):
    if selected_name is None:
        return ""
    df_binned = recategorize_y(df_clean.copy(), num_bins)
    person_row = df_binned[df_binned['name'] == selected_name]
    if person_row.empty:
        return f"No data found for {selected_name}"
    bin_name = person_row['golden_ratio_category'].values[0]
    return f"{selected_name} is in the '{bin_name}' bin"

# sophie callback
@app.callback(
    Output('pca-graph', 'figure'),
    Output('loadings-df', 'data'),
    Output('loadings-df', 'columns'),
    Input('model-selector', 'value')
)
def update_graph(selected_model):
    # 1. Run PCA
    features = feature_sets[selected_model]
    pc_df, loadings_df, eigvals = calculate_pca(features)
    
    loadings_df = loadings_df.reset_index().rename(columns={'index': 'Feature'})
    
    # 2. Base Plot
    fig = px.scatter(pc_df, x='PC1', y='PC2', 
                     hover_name='name',
                     hover_data=features + ['race', 'gender'], 
                     title=f"PCA Biplot: {selected_model}",
                     template='plotly_dark')

    # --- UPDATE DOTS TO GOLD ---
    fig.update_traces(marker=dict(color=theme['accent'], size=8))

    # 3. Add Arrows
    scale_factor = max(pc_df['PC1'].max(), pc_df['PC2'].max()) 
    
    for i, row in loadings_df.iterrows():
        var_name = row['Feature']
        x_end = row['PC1'] * scale_factor * 0.8
        y_end = row['PC2'] * scale_factor * 0.8
        
        fig.add_trace(go.Scatter(
            x=[0, x_end], y=[0, y_end],
            mode='lines+text',
            name=var_name,
            text=[None, var_name],
            textposition="top center",
            line=dict(width=3, color='white') # Arrows in White to contrast against Gold dots
        ))
        
        fig.add_annotation(
            x=x_end, y=y_end, ax=0, ay=0, xref="x", yref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='white'
        )

    # 4. Polish Layout
    fig.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        showlegend=False,
        paper_bgcolor=theme['background'],
        plot_bgcolor=theme['background'],
        font_color=theme['text']
    )
    
    display_df = loadings_df.round(4)
    columns = [{"name": i, "id": i} for i in display_df.columns]
    data = display_df.to_dict('records')

    return fig, data, columns

# sheyi's callback
@callback(
    Output('slider-output-container', 'children'),
    Input('bestK_slider', 'value'))
def update_output(value):
    pipe3 = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', KNeighborsClassifier(
            n_neighbors=value,
            weights="distance")
            )
    ])
    pipe3.fit(Xknn_train, yknn_train)
    yknn_pred = pipe3.predict(Xknn_test)
    
    knn_accacc = accuracy_score(yknn_test, yknn_pred)
    knn_bal_acc = balanced_accuracy_score(yknn_test, yknn_pred)
    val = f"Accuracy: {knn_acc:.3f}", f"Balanced accuracy: {knn_bal_acc:.3f}"

    return f"Accuracy: {knn_acc:.3f}   ", f"Balanced accuracy: {knn_bal_acc:.3f}"

# jills callbacks
@app.callback(
    Output('cluster-scatter-plot', 'figure'),
    Output('cluster-gender-plot', 'figure'),
    Output('cluster-race-plot', 'figure'),
    Input('tabs-on-top', 'value'),
    Input('k-value-slider', 'value')
)
def update_cluster_plots(tab_value, k_value):
    if tab_value == 'tab-6':
        fig_clusters, fig_cluster_gender, fig_cluster_race = k_means_func(k_value)
        return fig_clusters, fig_cluster_gender, fig_cluster_race
    empty_fig = {'data': [], 'layout': {'title': 'Select K-means tab to view'}}
    return empty_fig, empty_fig, empty_fig


# stephanie's callback 
@app.callback(
    [
        Output('lm-scatter', 'figure'),
        Output('lm-residuals', 'figure'),
        Output('model-summary', 'children'),
    ],
    [Input('regression-variable', 'value')]
)
def update_linear_tab(selected_feature):
    # 1. Safety Check
    if not selected_feature:
        return go.Figure(), go.Figure(), "Please select a feature."

    # 2. Prepare Single-Feature Data (Reshaping is crucial here)
    # We use the global Xlin_train/test variables defined earlier
    x_train_feat = Xlin_train[[selected_feature]].values
    x_test_feat  = Xlin_test[[selected_feature]].values
    y_train_vec  = ylin_train.values
    y_test_vec   = ylin_test.values

    # 3. Train Single-Feature Model
    sf_lr = LinearRegression()
    sf_lr.fit(x_train_feat, y_train_vec)
    
    # 4. Predictions & Metrics
    y_pred_test = sf_lr.predict(x_test_feat)
    residuals   = y_test_vec - y_pred_test
    
    r2_train = sf_lr.score(x_train_feat, y_train_vec)
    r2_test  = r2_score(y_test_vec, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_vec, y_pred_test))

    # 5. Create Line for Visualization (Min to Max of test data)
    x_range = np.linspace(x_test_feat.min(), x_test_feat.max(), 100).reshape(-1, 1)
    y_line  = sf_lr.predict(x_range)

    # --- PLOT 1: Scatter + Line ---
    scatter_fig = go.Figure()
    # Actual Test Points (Gold)
    scatter_fig.add_trace(go.Scatter(
        x=x_test_feat.ravel(), y=y_test_vec,
        mode='markers', name='Actual (Test)',
        marker=dict(color=theme['accent'], opacity=0.8, size=8)
    ))
    # Fitted Line (White)
    scatter_fig.add_trace(go.Scatter(
        x=x_range.ravel(), y=y_line,
        mode='lines', name='Fitted Line',
        line=dict(color='white', width=3)
    ))
    scatter_fig.update_layout(
        title=f"{selected_feature} vs Golden Score",
        xaxis_title=selected_feature,
        yaxis_title="Golden Score",
        template='plotly_dark',
        plot_bgcolor=theme['card_bg'],
        paper_bgcolor=theme['background'],
        font_color=theme['text']
    )

    # --- PLOT 2: Residuals ---
    resid_fig = go.Figure()
    resid_fig.add_trace(go.Scatter(
        x=x_test_feat.ravel(), y=residuals,
        mode='markers', name='Residuals',
        marker=dict(color=theme['accent'], opacity=0.8, size=8)
    ))
    # Zero line
    resid_fig.add_hline(y=0, line=dict(color='red', dash='dash'))
    resid_fig.update_layout(
        title=f"Residuals for {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Residual (Actual - Pred)",
        template='plotly_dark',
        plot_bgcolor=theme['card_bg'],
        paper_bgcolor=theme['background'],
        font_color=theme['text']
    )

    # --- SUMMARY TEXT ---
    # Using the GLOBAL RidgeCV results for the text summary
    coef_lines = [
        html.P(f"{feat}: {coef_map[feat]:.4f}", style={'margin': 0, 'color': theme['text']})
        for feat in predictive_cols
    ]
    
    summary = html.Div([
        html.H4("Multivariate Ridge Model Performance", style={'color': theme['accent']}),
        html.P(f"Global Test R²: {R2_TEST:.3f}", style={'fontWeight': 'bold'}),
        html.P(f"Global Test RMSE: {RMSE_TEST:.3f}"),
        html.P(f"Best Alpha: {alpha_selected:.3f}"),
        html.Hr(style={'borderColor': theme['accent']}),
        html.P("Feature Coefficients:", style={'color': theme['accent']}),
        html.Div(coef_lines, style={'paddingLeft': '20px'}), 
        html.H5("Conclusion:", style={'color': theme['accent'], 'marginBottom': '6px'}),
            html.P(
                "An L2 regularization is used on the linear model to stabilize the coefficients due to the raw facial measurements being highly correlated. "
                "The single feature scatter plots above show the predictive relationship between golden_score and an individual predictor. "
                "The scatterplots reveal that individual predictors have weak predictive power. However, when using a multivariate model,predictive accuracy increases. "
                "A multivariate model results in a R² value of 0.926 and RMSE of about 0.94, indicating that a good proportion of variance in golden_score is explained by a multivariate linear model. "
                "Therefore, raw individual measurements alone are insufficient in predicting golden ratio.",
                style={'color': theme['text'], 'fontSize': '16px', 'lineHeight': '1.6'}
        )
        ],
        style={'padding':'16px','backgroundColor': theme['card_bg'],'border': f'1px solid {theme["accent"]}','borderRadius':'8px','marginTop':'20px'})

    return scatter_fig, resid_fig, summary

if __name__ == '__main__':
    app.run(debug=True)