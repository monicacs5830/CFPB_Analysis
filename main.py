# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + id="uYFMpGSLvSDr"
# # !pip install gcsfs
# # !pip install google-cloud-bigquery

# + id="79DIQqEhe1ec"
# # !pip install dash-bootstrap-components

# + id="vh5Xmy393mZ5" outputId="e1723ff7-75a3-469d-fbc2-137eb2a1904a" colab={"base_uri": "https://localhost:8080/"}
 # !pip install dash

# + id="ySkmLgAw26zR"
# # !pip install gcsfs


# + id="Yu3-zwfMvU-n"
# importing the necessary libraries for data manipulation and visualization
from google.cloud import bigquery
import gcsfs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dependencies import Input, Output, State
import random
import plotly.express as px

# + id="PDh6K4h2wLeL"
# Authenticating with the Google Cloud account and setting up the BigQuery client:
import os
import subprocess

# URL to the raw GitHub file
url = "https://raw.githubusercontent.com/monicacs5830/CFPB_Complaints_Analysis/main/data608-391503-e77b21b1d01c.json"

# Use wget to download the file
subprocess.run(["wget", url, "-O", "key.json"])

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './key.json'

# Now you can create your BigQuery client
client = bigquery.Client()

# + id="kCHKzJZTvmR9"
# Accessing the data:
dataset_ref = client.dataset("cfpb_complaints", project="bigquery-public-data")
cfpb_complaints_table = dataset_ref.table('complaint_database')
df_complaints = client.get_table(cfpb_complaints_table)


# + id="t-bdNzK7vuAy"
# Using BigQuery to pull data into a Pandas DataFrame for initial exploration
# query = """
# SELECT *
# FROM `bigquery-public-data.cfpb_complaints.complaint_database`
# LIMIT 100000
# """
# #
# df_complaints = client.query(query).to_dataframe()


# + colab={"base_uri": "https://localhost:8080/"} id="EWWbyFygxRB6" outputId="fea91e3c-8729-4cb1-9f53-3e4594bf2058"
# list of columns to check for nulls
columns_to_check = [
    "subissue",
    "subproduct",
    "consumer_complaint_narrative",
    "company_public_response",
    "state",
    "zip_code",
    "tags",
    "consumer_disputed",
    "consumer_consent_provided",
    "company_response_to_consumer"
]

# SQL query string
missing_data_query = ",\n".join(f"COUNTIF({col} IS NULL) AS {col}_missing_count" for col in columns_to_check)


missing_data_query = f"""
SELECT
  {missing_data_query}
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
"""

# querying the data
missing_data = client.query(missing_data_query).to_dataframe()

# printing the data
print(missing_data)


# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="c8_0HSyLKKOe" outputId="c51c4371-d9b5-4372-ff32-5830ef27856c"
# 1.How are complaints distributed by product and sub-product?
#Is there any variation in this distribution by state?

# Query for product counts
query_product_counts = """
    SELECT product, COUNT(product) as Counts_Complaints
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE subproduct IS NOT NULL
    GROUP BY product
    ORDER BY Counts_Complaints DESC
"""
query_job_product_counts = client.query(query_product_counts)
products_df_pandas = query_job_product_counts.to_dataframe()

# Get the unique products that have a subproduct associated with them
products = products_df_pandas['product'].unique().tolist()

# Now we ensure the products and subproducts are sorted in descending order
# by complaint count and that we only include products with subproducts.
query1a = """
    SELECT product, subproduct, state, COUNT(state) as Counts_Complaints
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE subproduct IS NOT NULL
    GROUP BY product, subproduct, state
    ORDER BY COUNT(state) DESC
"""
# Execute the query
query_job = client.query(query1a)
product_subproduct_state_df = query_job.to_dataframe()

# Get the mapping from product to subproduct
product_to_subproduct = product_subproduct_state_df.groupby('product')['subproduct'].unique().apply(list).to_dict()

# Get the mapping from product to subproduct
product_to_subproduct = product_subproduct_state_df.groupby('product')['subproduct'].unique().apply(list).to_dict()

abbr_mapping = {
    'Credit reporting, credit repair services, or other personal consumer reports': 'CR, CRS, or OPC Reports',
    'Debt collection': 'Debt Collection',
    'Mortgage': 'Mortgage',
    'Credit card or prepaid card': 'CC or PC',
    'Checking or savings account': 'CoSA',
    'Credit reporting': 'Credit Reporting',
    'Credit card': 'Credit Card',
    'Bank account or service': 'BAS',
    'Student loan': 'Student Loan',
    'Money transfer, virtual currency, or money service': 'MT, VC, or MS',
    'Vehicle loan or lease': 'VLL',
    'Consumer Loan': 'Consumer Loan',
    'Payday loan, title loan, or personal loan': 'PL, TL, or PL',
    'Payday loan': 'Payday Loan',
    'Money transfers': 'Money Transfers',
    'Prepaid card': 'Prepaid Card',
    'Other financial service': 'OFS',
    'Virtual currency': 'Virtual Currency'
}

# Apply abbreviation mapping to 'product' column
products_df_pandas['product_abbr'] = products_df_pandas['product'].map(abbr_mapping)

# Import plotly colors
from plotly.express.colors import qualitative

# Plot the graph
fig1a = px.bar(
    products_df_pandas,
    x='product_abbr',
    y='Counts_Complaints',
    color='product',
    #color_discrete_sequence=qualitative.Prism
)
fig1a.update_layout(
    autosize=True,
    title_text = 'Distribution of Complaint Types (Product)'
)


# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="dPgiPNvAWeXN" outputId="18513c9f-cb83-4506-a081-828b65c9ca05"
# Query
query2a = """
    SELECT IFNULL(issue, 'Unknown') as issue, COUNT(issue) as count
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    GROUP BY issue
    ORDER BY count DESC
"""
# Execute the query
query_job = client.query(query2a)
issue_df = query_job.to_dataframe()

# Query for issue, subissue
query_subissue = """
    SELECT IFNULL(issue, 'Unknown') as issue, IFNULL(subissue, 'Unknown') as subissue, COUNT(issue) as count
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    GROUP BY issue, subissue
    ORDER BY count DESC
"""
query_job_subissue = client.query(query_subissue)
issue_subissue_df = query_job_subissue.to_dataframe()

# Count and sort issues
issue_counts = issue_subissue_df.groupby('issue')['count'].sum()
issue_counts_sorted = issue_counts.sort_values(ascending=False)
issues = issue_counts_sorted.index.tolist()

# Mapping from issue to subissues
issue_to_subissue = issue_subissue_df.groupby('issue')['subissue'].unique().apply(list).reindex(issues).to_dict()

# Get top 10 issues
top_issues = issue_counts_sorted.head(10)

# Filter data for top issues
issue_subissue_df_top = issue_subissue_df[issue_subissue_df['issue'].isin(top_issues.index)].nlargest(15, 'count')

# Abbreviations for issue names
abbreviations = {
    'Incorrect information on your report': 'Incorrect Info',
    'Problem with a credit reporting company\'s investigation into an existing problem': 'Investigation Problem',
    'Improper use of your report': 'Improper Use',
    'Loan modification,collection,foreclosure': 'Loan Modification',
    'Attempts to collect debt not owed': 'Unowed Debt Collection',
    'Loan servicing, payments, escrow account': 'Loan Servicing',
    'Trouble during payment process': 'Payment Trouble',
    'Written notification about debt': 'Debt Notification',
}

# Applying the abbreviations to the dataframe
issue_subissue_df_top['issue_abbrev'] = issue_subissue_df_top['issue'].map(abbreviations)

# Creating the plot with abbreviations
fig2b = px.bar(issue_subissue_df_top, y='issue_abbrev', x='count', color='subissue', orientation='h',
             title='Counts of Complaints by Issue and Subissue',
             labels={'count':'Count of Complaints', 'issue_abbrev':'Issue'},
             hover_data=['issue'],
             #color_discrete_sequence=qualitative.Prism
               ) # include the original issue as a hover-over

fig2b.update_layout(yaxis={'categoryorder': 'total ascending'})
# fig2b.show()


# + id="BQXeR3e6Yzz0"
#display(issue_query.head(10))
# display(issue_subissue_query.head(10))

# + id="AQMBVv9FZobQ"
#3. Submission Method Impact
#Is there any relationship between the method used to submit the complaint
#and the response from the company or the outcome of the dispute?

# Query
query3 = """
    SELECT submitted_via, company_response_to_consumer, COUNT(*) as Count_Response
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    GROUP BY submitted_via, company_response_to_consumer
    ORDER BY Count_Response DESC
"""
# Execute the query
query_job = client.query(query3)
method_df_pd = query_job.to_dataframe()

# Plotting
fig3 = px.bar(method_df_pd, x='submitted_via', y='Count_Response',
             color='company_response_to_consumer',
             title='Counts of Company Responses by Submission Method',
             labels={'Count_Response':'Count of Responses',
                     'submitted_via':'Submission Method',
                     'company_response_to_consumer':'Company Response'})
# fig3.show()



# + colab={"base_uri": "https://localhost:8080/"} id="E0skmiUiXvRf" outputId="94d450b2-0c67-4186-f124-91d219e5f32d"
# checking association using contingency table
from scipy.stats import chi2_contingency

# Create a cross-tabulation (contingency table)
contingency_table = pd.crosstab(method_df_pd['submitted_via'], method_df_pd['company_response_to_consumer'])

# Perform the Chi-Square Test of Independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("chi2 statistic", chi2)
print("p-value", p)

# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="Hn9UiFvZhL8X" outputId="916b4341-96a9-4035-820a-8300ce7f786a"
# Data preparation and plot creation for page 4
# 4a: analyzing total complaints over time
# Query
query4a = """
    SELECT DATE(date_received) as date_received, COUNT(*) as count
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    GROUP BY date_received
    ORDER BY date_received
"""
# Execute the query
query_job = client.query(query4a)
complaints_over_time = query_job.to_dataframe()

# Monthly moving average
complaints_over_time['monthly_mva'] = complaints_over_time['count'].rolling(window=30).mean()

# Base line plot
fig4a = go.Figure()
fig4a.add_trace(go.Scatter(x=complaints_over_time['date_received'], y=complaints_over_time['count'],
                           mode='lines', name='Complaints', line=dict(color='#FA9A85', width=0.5, dash='dash')))
fig4a.add_trace(go.Scatter(x=complaints_over_time['date_received'], y=complaints_over_time['monthly_mva'],
                           mode='lines', name='Monthly Moving Average', line=dict(color='darkred', width=2)))
fig4a.update_layout(title='Trend in complaints received over time',
                   xaxis_title='Date Received',
                   yaxis_title='Number of Complaints')

#fig4a.show()





# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="HiO9TU-14ZuU" outputId="9ad5b668-67f4-4411-a13c-305f5dadb6f3"
# 4b: analyzing types of company responses over time
# Query
query4b = """
    SELECT DATE(date_received) as date_received, company_response_to_consumer, COUNT(*) as count
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE company_response_to_consumer IS NOT NULL
    GROUP BY date_received, company_response_to_consumer
    ORDER BY date_received
"""
# Execute the query
query_job = client.query(query4b)
responses_over_time = query_job.to_dataframe()

# Pivot table for better data structure
pivot_response_df = responses_over_time.pivot(index='date_received', columns='company_response_to_consumer', values='count').fillna(0)
pivot_response_df_mva = pivot_response_df.rolling(window=30).mean()  # Monthly moving average

# Create the plot
fig4b = go.Figure()
for col in pivot_response_df_mva.columns:
    if str(col).lower() != 'nan':
        fig4b.add_trace(go.Scatter(x=pivot_response_df_mva.index, y=pivot_response_df_mva[col], mode='lines', name=col))
fig4b.update_layout(title="Trend in Company Responses Over Time (Moving Average)",
                   xaxis_title="Date Received",
                   yaxis_title="Number of Responses (Moving Average)",
                   legend_title="Company Response Type")

# fig4b.show()


# + id="9U7zMGPpY1zv"
# Query
query4c = """
    SELECT DATE(date_received) as date_received, product, COUNT(*) as count
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE product IS NOT NULL
    GROUP BY date_received, product
    ORDER BY date_received
"""
# Execute the query
query_job = client.query(query4c)
product_complaints_over_time = query_job.to_dataframe()

# Pivot table for better data structure
pivot_df = product_complaints_over_time.pivot(index='date_received', columns='product', values='count').fillna(0)
pivot_df_mva = pivot_df.rolling(window=30).mean()  # Monthly moving average

# # Create the plot
# fig4c = go.Figure()
# for col in pivot_df_mva.columns:
#     if str(col).lower() != 'nan':
#         fig4c.add_trace(go.Scatter(x=pivot_df_mva.index, y=pivot_df_mva[col], mode='lines', name=col))
# fig4c.update_layout(title="Trend in Complaints by Product Over Time (Moving Average)",
#                    xaxis_title="Date Received",
#                    yaxis_title="Number of Complaints (Moving Average)",
#                    legend_title="Product")

# fig4c.show()


# + id="2OwryrWd5jbB"
# 4d: analyzing complaints by issue over time
# Query
query4d = """
    SELECT EXTRACT(YEAR FROM date_received) as year, issue, COUNT(*) as count
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    GROUP BY year, issue
    ORDER BY year, count DESC
"""
# Execute the query
query_job = client.query(query4d)
issue_complaints_over_time = query_job.to_dataframe()

# Replace missing issue values with 'Unknown'
issue_complaints_over_time['issue'] = issue_complaints_over_time['issue'].fillna('Unknown')

# Getting the issue with max complaints for each year
issue_complaints_over_time = issue_complaints_over_time.loc[issue_complaints_over_time.groupby('year')['count'].idxmax()]

# Creating the plot
fig4d = px.bar(issue_complaints_over_time, x='year', y='count',
             color='issue',
             title='Issue with the highest number of complaints by year',
             labels={'year':'Year', 'count':'Number of Complaints', 'issue':'Issue Type'})

# fig4d.show()



# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="gNmV5kYTZQOn" outputId="937590ec-f687-496d-aed6-45bdb31102f4"
#Data Preparation for Page 5
# Question 5b: Outcome Impact: Does the resolution method, whether monetary or nonmonetary relief, influence the likelihood of a customer dispute?

# Query
query5a = """
    SELECT company_response_to_consumer, COUNT(*) as count
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE company_response_to_consumer IS NOT NULL
    GROUP BY company_response_to_consumer
"""
# Execute the query
query_job = client.query(query5a)
response_counts_pandas_df = query_job.to_dataframe()

# Replace missing issue values with 'Unknown'
response_counts_pandas_df['company_response_to_consumer'] = response_counts_pandas_df['company_response_to_consumer'].fillna('Unknown')

# Creating the plot
fig5a = px.pie(response_counts_pandas_df,
             values='count',
             names='company_response_to_consumer',
             title='Composition of Different Company Response Types')

# Changing the pie chart to donut chart by adding a hole parameter
fig5a.update_traces(hole=.3)

# fig5a.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 542} id="F4M0d5KcZrlI" outputId="839aa8d5-6576-44bb-d76f-fa070cf7431b"
# 5b: How does Company's response impact the dispute rate
# List of required responses
required_responses = [
    'Closed',
    'Closed with explanation',
    'Closed with monetary relief',
    'Closed with non-monetary relief'
]

# Query
query5b = """
    SELECT company_response_to_consumer as Response, consumer_disputed as Dispute, COUNT(*) as Count_response
    FROM `bigquery-public-data.cfpb_complaints.complaint_database`
    WHERE company_response_to_consumer IS NOT NULL
    AND company_response_to_consumer IN UNNEST(@required_responses)
    GROUP BY Response, Dispute
"""
query_params = [
    bigquery.ArrayQueryParameter("required_responses", "STRING", required_responses)
]

# Execute the query
job_config = bigquery.QueryJobConfig()
job_config.query_parameters = query_params
query_job = client.query(query5b, job_config=job_config)

filtered_df = query_job.to_dataframe()

# display(filtered_df)

# Filter out NA in disputes
filtered_df = filtered_df[filtered_df['Dispute'].notna()]

# Change boolean values to string representation
filtered_df['Dispute'] = filtered_df['Dispute'].apply(lambda x: 'True' if x else 'False')

# Plotting
fig5b = px.pie(filtered_df,
             values='Count_response',
             names='Dispute',
             color='Dispute',
             title='Percentage of Disputes and Non-Disputes for Each Response Type',
             facet_col='Response',
             facet_col_wrap=2,
             color_discrete_map={'True': '#DC3912', 'False': '#2CA02C'})

fig5b.update_traces(textinfo='percent+label',
                    insidetextorientation='radial',
                    textposition='inside')

# fig5b.show()



# + colab={"base_uri": "https://localhost:8080/"} id="d_J_OMXbd59t" outputId="3b6e646c-51e7-4509-ea6c-79b9814a93b1"
# Data for Page 6
import pandas as pd
import pickle

# Load the pickled model
with open('Complement_NB_model.pkl', 'rb') as file:
    Complement_NB = pickle.load(file)

# Load the test dataset
Narrative_test_set = pd.read_csv('https://raw.githubusercontent.com/AliHaghighat1/Consumer-Complaints/main/Narrative_test.csv')


# + colab={"base_uri": "https://localhost:8080/", "height": 671} id="PXhshYrkeojn" outputId="4aa96846-264a-4f84-ea5e-7ca7004e3184"
# Initializing Dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO], suppress_callback_exceptions=True)
server = app.server
# Button style with padding
button_style = {'margin': '20px', 'margin-left': '20px', 'margin-right': '20px'}
# Heading stype with padding
heading_style = {'margin': '15px', 'margin-left': '10px', 'margin-right': '10px'}

# defining the header
# header = dbc.NavbarSimple(
#    children=[
#        dbc.NavItem(dbc.NavLink("Consumer Financial Protection Bureau", href="https://www.consumerfinance.gov")),
#    ],
#     brand="Financial Complaints Analysis",
#    brand_href="/",
#    color="light",
#    dark=False,
#    className="mb-4",
#)

# Defining the footer
footer = dbc.Container(
    [
        html.Hr(),  # Add a horizontal line
        dbc.Row(
            [
                html.P("For More Information Click on the GitHub Link here:", style={'textAlign': 'center'}),
                dbc.Col(
                    html.A(
                        html.Img(src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png', height="40px"),
                        href="https://github.com/monicacs5830/CFPB_Complaints_Analysis"
                    ),
                    style={'textAlign': 'center'}
                ),
            ],
            align="center",  # Center the contents vertically
            justify="center",  # Center the contents horizontally
        ),
    ],
    fluid=True
)

# defining the sidebar
sidebar = dbc.Nav(
    [
        dbc.NavLink("Home", href="/", active="exact"),
        dbc.NavLink("Products and Subproducts Distribution", href="/page-1", active="exact"),
        dbc.NavLink("Issues and Subissues Distribution", href="/page-2", active="exact"),
        dbc.NavLink("Complaints by Submission Method", href="/page-3", active="exact"),
        dbc.NavLink("Trends in Complaints in CFPB", href="/page-4", active="exact"),
        dbc.NavLink("Outcome Impact Analysis", href="/page-5", active="exact"),
        dbc.NavLink("Prediction of Company's Response", href="/page-6", active="exact"),
    ],
    vertical="md",
    pills=True,
    id='sidebar',
)
# definind the navbar
navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    dbc.NavbarBrand(
                        html.Img(src="https://cfpb.github.io/design-system/images/uploads/logo_background_photo.png",
                                 style={'width':'130%', 'height':'90%'}),
                        className="ml-2",
                        style={'height': '90%', 'width':'130%'}
                    ),
                    width=12,
                    align="start"
                ),
                dbc.Col(
                    dbc.Button(
                        "Menu",
                        id="btn_sidebar",
                        n_clicks=0,
                        className="mr-2",
                        style={'margin':'20px'},
                    ),
                    width="auto",
                    align="end"
                ),
            ],
            align="center",
            justify="between",
        ),
        fluid=True,
    ),
    color="light",  # Changed to light
    dark=False,  # Set to false for a light navbar
    className="mb-4",
)




# Put sidebar inside a dbc.Collapse
sidebar_collapse = dbc.Collapse(sidebar, id="collapse")

# Layouts

# Using BigQuery to pull data into a Pandas DataFrame for initial exploration
query = """
SELECT *
FROM `bigquery-public-data.cfpb_complaints.complaint_database`
LIMIT 5
"""
#
sample_data = client.query(query).to_dataframe() # limit data to 5 rows and convert to pandas

# Defining the homepage with introduction, sample data and buttons
homepage = html.Div([
    html.H1(" Understanding Financial Complaint Outcomes : A Data-Driven Approach", style= heading_style),
#     html.Br(),
    html.Img(src='https://www.centralbank.ie/images/default-source/consumer-hub/explainers---banner-images/complaints-banner.jpg?sfvrsn=4',
             style={'display':'block','width':'60%', 'height':'20%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    html.Br(),
    html.P("""
    The financial industry plays a key role in supporting personal and business financial goals.
    Like any sector, it sees its fair share of customer dissatisfaction, which, when analyzed, can offer crucial feedback and insights into customer expectations and problem areas.
    This project aims to analyze financial product complaints, predict their outcomes, and uncover insights into consumer behavior and possible improvements for financial services.
    """),
    html.H2("Dataset"),
    html.Br(),
    html.Br(),
    html.P("""
    We use the Consumer Financial Protection Bureau (CFPB) Complaint Database, hosted on Google's BigQuery public datasets.
    This 2.15GB dataset consists of over 3.4 million rows of data on consumer complaints related to financial products and services reported to the CFPB from 2011 to 2023.
    Key variables include 'date_received', 'product', 'issue', 'consumer_complaint_narrative', 'company_name', 'state', 'company_response_to_consumer', 'timely_response', and 'consumer_disputed'.
    The latter two will serve as response variables for complaint distribution and satisfaction analysis.
    """),
    html.Br(),
    html.H5("Below is a snapshot of our data:"),
    dbc.Card(
        dbc.CardBody(
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in sample_data.columns],
                data=sample_data.to_dict('records'),
                style_cell={'color': 'black'} ,
                style_table={'overflowX': 'auto'},  # Enable horizontal scrolling
            ),
        ), style={'margin': '20px'}, className= "bg-primary mb-3"
    ),
    html.Br(),
    dbc.Container(
        dbc.Row(
            html.Div(className='nav-buttons', children=[
                dbc.Button("Next Section", color="primary", href='/page-1', className="mr-1"),
            ], style={'textAlign': 'center', 'margin': '20px'}), justify='center'
        ), fluid=True
    ),
])
# Layout for page 1
page_1 = html.Div([
#     dcc.Link('Go to Home', href='/'),
    html.Br(),
    html.H3("Complaint Distribution"),
    html.Br(),
    html.H4("How are complaints distributed by product and sub-product? Is there any variation in this distribution by state?"),
    html.Br(),
    dcc.Graph(figure=fig1a),
    html.Br(),
    html.P("The graph illustrates complaint numbers for different financial product categories. 'Credit reporting, credit repair services, or other personal consumer reports' received the most complaints (1.6 million), indicating consumer issues. 'Debt collection' followed with over 0.4 million complaints, reflecting challenges faced with debt collectors. 'Mortgage' had just below 0.4 million complaints, unsurprising due to its significant financial commitment. Other categories had fewer complaints, all below 0.2 million counts, suggesting fewer issues compared to credit reporting, debt collection, and mortgages. Interestingly, 'virtual currency' had the fewest complaints, implying relative consumer satisfaction. The graph offers valuable insights for companies and regulators to enhance consumer experiences and address potential issues effectively."),
    html.Br(),
    html.H4("Interactive Choropleth Map of Complaint Types by State"),
    html.Br(),
    html.H5("Select a Product:"),
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': product, 'value': product} for product in products],
        value=products[0],
        style={'color': '#000', 'background-color': '#fff'}
    ),
    html.Br(),
    html.H5("Select a Subproduct:"),
    dcc.Dropdown(
        id='subproduct-dropdown',
        style={'color': '#000', 'background-color': '#fff'}
    ),
    html.Br(),
    dcc.Graph(id='choropleth-graph'),
    html.Br(),
    html.Br(),
#     dbc.Button("Previous Section", color="primary", href='/page-4', className="mr-1", style=button_style),
    html.Br(),
    dbc.Container(
        dbc.Row(
            html.Div(className='nav-buttons', children=[
#                 dbc.Button("Previous Section", color="primary", href='/page-1', className="mr-1"),
                dbc.Button("Home", color="primary", href='/', className="mr-1"),
                dbc.Button("Next Section", color="primary", href='/page-2', className="mr-1", style={'margin-left': '20px'}),
            ], style={'textAlign': 'center', 'margin': '20px'}), justify='center'
        ), fluid=True
    ),
])

second_question_key_findings = """
Key Findings:
- "Incorrect information on one's report" was the most common issue with 800,000 complaints, indicating credit report inaccuracies.
- The second most common issue was related to investigations by credit reporting companies, with over 400,000 complaints, highlighting concerns about their effectiveness.
- "Improper use of the report" had 300,000 complaints, showing credit reports are sometimes used inappropriately.
- "Attempting to collect debt not owed" received over 100,000 complaints, indicating collection attempts for debts consumers didn't owe.
- "Written notification about debt" had the fewest complaints, less than 100,000.

Sub-issues in "incorrect information on the report":
- "Information belonging to someone else" had over 500,000 complaints.
- "Incorrect account information" and "incorrect status" had 99,000 and 98,000 complaints, respectively.
- "Incorrect personal information" received the least complaints with only 50,000 cases.

In summary, the analysis shows the need for addressing credit report inaccuracies and improving practices by credit reporting companies.
"""

# Layout for page 2
page_2 = html.Div([
#     dcc.Link('Go to Home', href='/'),
    html.Br(),
    html.H3("Issue Prevalence"),
    html.Br(),
    html.H4("What are the most frequently mentioned issues and sub-issues in the complaints?"),
    html.Br(),
    html.H4("Interactive Plot for Distribution of Issue Types"),
    html.Br(),
    html.H5("Select an Issue:"),
    dcc.Dropdown(
        id='issue-dropdown',
        options=[{'label': issue, 'value': issue} for issue in issues],
        value=issues[0],
        style={'color': '#000', 'background-color': '#fff'}
    ),
    dcc.Graph(id='bar-graph'),
    dcc.Graph(figure=fig2b),
    html.Br(),
    html.P(second_question_key_findings, style={'font-size': '18px'}),
    html.Br(),
    html.Br(),
    dbc.Container(
        dbc.Row(
            html.Div(className='nav-buttons', children=[
                dbc.Button("Previous Section", color="primary", href='/page-1', className="mr-1"),
                dbc.Button("Home", color="primary", href='/', className="mr-1", style={'margin-left': '20px'}),
                dbc.Button("Next Section", color="primary", href='/page-3', className="mr-1", style={'margin-left': '20px'}),
            ], style={'textAlign': 'center', 'margin': '20px'}), justify='center'
        ), fluid=True
    ),
])

#Layout for page 3
page_3 = html.Div([
#     dcc.Link('Go to Home', href='/'),
    html.Br(),
    html.H3("Submission Method Impact"),
    html.Br(),
    html.H4("Is there any relationship between the method used to submit the complaint and the response from the company or the outcome of the dispute? "),
    html.Br(),
    dcc.Graph(figure=fig3),
    html.Br(),
    html.P(
        """
        According to the graph above, it illustrates the number of company responses to complaints, categorized by different methods of complaint submission. The submission method that received the highest number of company responses was through the web.
        For all types of submission methods, the most common company response was to close the complaints with an explanation. This indicates that a significant proportion of complaints were resolved by providing consumers with a detailed explanation of the actions taken or the reasons behind the issues raised.
        The next most frequent company response was to close the complaints with non-monetary relief. This suggests that many complaints were resolved by offering remedies or solutions that did not involve monetary compensation but aimed to address the concerns and provide some form of resolution.
        Responses categorized as "in progress" or "closed with monetary relief" followed closely after non-monetary relief. This implies that some complaints required ongoing attention or were ultimately resolved with financial compensation to the affected consumers.
        However, there were notable issues with certain submission methods. Specifically, complaints submitted via referral, phone, postal mail, or fax had low numbers of responses, and web referral or email complaints were not responded to at all by the company. This highlights a concerning lack of responsiveness from companies to complaints submitted through these channels.
        In summary, the graph reveals that web submissions were the most common method for consumers to submit complaints, and most of these complaints received responses in the form of explanations or non-monetary relief. Nevertheless, there was a need for improvement in addressing complaints submitted via web referral or email, as companies failed to respond to those submissions.
        """
    ),
    html.Br(),
    html.Br(),
    html.H4("Chi-Square Test of Independence to determine whether there is a significant relationship between the method used to submit complaints (submitted_via) and the response from the company (company_response_to_consumer)."),
    html.Br(),
    html.P("Null Hypothesis: There is no significant relationship between the two variables"),
    html.Br(),
    html.P("Alternative Hypothesis: There is a significant relationship."),
    html.Br(),
    html.Div(id='chi-square-test'),
    html.Br(),
    html.Br(),
    dbc.Container(
        dbc.Row(
            html.Div(className='nav-buttons', children=[
                dbc.Button("Previous Section", color="primary", href='/page-2', className="mr-1"),
                dbc.Button("Home", color="primary", href='/', className="mr-1", style={'margin-left': '20px'}),
                dbc.Button("Next Section", color="primary", href='/page-4', className="mr-1", style={'margin-left': '20px'}),
            ], style={'textAlign': 'center', 'margin': '20px'}), justify='center'
        ), fluid=True
    ),
])

complaints_over_time['date_received'] = pd.to_datetime(complaints_over_time['date_received'])
complaints_over_time = complaints_over_time[complaints_over_time['date_received'].dt.year >= 2012]

# Layout for page 4
page_4_layout = html.Div([
    html.Br(),
    html.H3("Trend Identification"),
    html.Br(),
    html.H4("Can we discern any trends in how the CFPB handles cases or how companies respond over time? Are there unusually high instances of responses for any specific product or issue category?"),
    html.Br(),
    html.H4("Analyzing total complaints over time"),
    dcc.Slider(
    id='year-slider',
    min=complaints_over_time['date_received'].min().year,
    max=complaints_over_time['date_received'].max().year,
    value=complaints_over_time['date_received'].min().year,
    marks={str(year): str(year) for year in complaints_over_time['date_received'].dt.year.unique()},
    step=None
    ),
    dcc.Graph(
        id='graph_4a'
    ),
#     dcc.Link('Go to Home', href='/'),
#     html.Br(),
#     html.H3("Trend Identification"),
#     html.Br(),
#     html.H4("Can we discern any trends in how the CFPB handles cases or how companies respond over time? Are there unusually high instances of responses for any specific product or issue category?"),
#     html.Br(),
#     html.H4("Analyzing total complaints over time"),
#     dcc.Graph(
#         id='graph_4a',
#         figure={
#             'data': [
#                 go.Scatter(x=complaints_over_time['date_received'], y=complaints_over_time['count'],
#                            mode='lines', name='Actual Frequency of Complaints'),
#                 go.Scatter(x=complaints_over_time['date_received'], y=complaints_over_time['monthly_mva'],
#                            mode='lines', name='Monthly Moving Average of Complaints')
#             ],
#             'layout': go.Layout(title='Trend in complaints received over time')
#         }
#     ),
    html.Br(),
    html.P("""The above graph
    portrays a consistent pattern of complaints over time, followed by a significant increase in recent years, reaching a peak in 2023.
    This trend could indicate various factors influencing consumer experiences or a potential rise in consumer awareness and reporting during that period.
    Further analysis would be required to understand the drivers behind this notable shift in complaint volumes."""),
    html.Br(),
    html.H4("Analyzing Company Responses Over Time"),
    dcc.Graph(figure=fig4b),
    html.Br(),
    html.P("""The graph above highlights the trends in company responses to consumer complaints over the years.
    While some response categories remained relatively stable, there were significant changes in others, indicating evolving practices and a potential focus on more thorough explanations and non-monetary remedies in recent times.
    The data offers valuable insights into how companies addressed consumer complaints and grievances during this decade-long period."""),
    html.Br(),
    html.H4("Analyzing Complaints by Product Types Over Time"),
    dcc.Dropdown(
        id='dropdown_4c',
        options=[
            {'label': i, 'value': i} for i in pivot_df_mva.columns
        ],
        value=pivot_df_mva.columns[0],
        style={'color': '#000', 'background-color': '#fff'}
    ),
    dcc.Graph(
        id='graph_4c'
    ),
    html.H4("Analyzing issues over time"),
    dcc.Graph(figure=fig4d),
    html.Br(),
    html.P("""The graph indicates that concerns related to credit reporting, such as incorrect information on reports, have consistently dominated the top complaint categories for consumers over the years.
    The data also suggests a notable increase in the number of complaints in recent years, particularly regarding issues with credit reports, signaling a growing area of consumer frustration and attention."""),
    html.Br(),
    dbc.Container(
        dbc.Row(
            html.Div(className='nav-buttons', children=[
                dbc.Button("Previous Section", color="primary", href='/page-3', className="mr-1"),
                dbc.Button("Home", color="primary", href='/', className="mr-1", style={'margin-left': '20px'}),
                dbc.Button("Next Section", color="primary", href='/page-5', className="mr-1", style={'margin-left': '20px'}),
            ], style={'textAlign': 'center', 'margin': '20px'}), justify='center'
        ), fluid=True
    ),
])

# Layout for Page 5
page_5_layout = html.Div([
#     dcc.Link('Go to Home', href='/'),
    html.Br(),
    html.H3("Outcome Impact"),
    html.Br(),
    html.H4("Does the resolution method, whether monetary or nonmonetary relief, influence the likelihood of a customer dispute?"),
    html.Br(),
    dcc.Graph(id='page-5-graph-a', figure=fig5a),
    html.Br(),
    html.P("""
    The above graph indicates that the majority of consumer complaints received thorough attention through explanations from companies, while a notable portion received non-monetary relief.
    The relatively low percentage of complaints closed with monetary relief suggests that financial compensation was not the primary resolution approach.
    Additionally, the presence of complaints still "in progress" underscores the ongoing efforts to address consumer concerns.
    Overall, the data provides valuable insights into how companies handled various types of consumer complaints."""),
    html.Br(),
    dcc.Graph(id='page-5-graph-b', figure=fig5b),
    html.Br(),
    html.P("""From the above graph, it can be observed that complaints resolved with monetary relief have a lower likelihood of resulting in a dispute compared to other resolution methods (11.3%).
    Conversely, complaints closed with explanation or simply closed have relatively higher dispute rates (24.2% and 26.5%, respectively).
    """),
    html.Br(),
    dbc.Container(
        dbc.Row(
            html.Div(className='nav-buttons', children=[
                dbc.Button("Previous Section", color="primary", href='/page-4', className="mr-1"),
                dbc.Button("Home", color="primary", href='/', className="mr-1", style={'margin-left': '20px'}),
                dbc.Button("Next Section", color="primary", href='/page-6', className="mr-1", style={'margin-left': '20px'}),
            ], style={'textAlign': 'center', 'margin': '20px'}), justify='center'
        ), fluid=True
    ),
])

# Layout for Page 6
page_6_layout = html.Div([
    dbc.Container(
        dbc.Row(
            [
                html.H3("The Company's Response Prediction Based On the Consumer Narrative", className="text-center"),
                html.P("The narratives are selected randomly from the test dataset when you click on Generate Prediction button below", className="text-center"),
                dbc.Button("Generate Prediction", id="prediction-button", className="mr-1", style={'margin': '20px'}),
            ], className="py-5 text-center"
        ), fluid=True
    ),
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Consumer Narrative", className="card-title"),
                                ),
                                dbc.CardBody(
                                    [
                                        html.P(id="narrative", className="card-text"),
                                    ]
                                )
                            ], className="text-black bg-light mb-3"
                        ), md=4
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Predicted Response", className="card-title"),
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5(id="prediction", className="card-text"),
                                    ]
                                )
                            ], className="text-black bg-light mb-3"
                        ), md=4
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.H4("Probability Of The Response", className="card-title"),
                                ),
                                dbc.CardBody(
                                    [
                                        html.H5(id="prob_max", className="card-text"),
                                    ]
                                )
                            ], className="text-black bg-light mb-3"
                        ), md=4
                    ),
                ]
            ),
        ], fluid=True
    ),
    #dbc.Button("Generate prediction", id="prediction-button", className="mr-1", style={'margin': '20px'}),
    html.Br(),
    html.Br(),
    dbc.Container(
        dbc.Row(
            html.Div(className='nav-buttons', children=[
                dbc.Button("Previous Section", color="primary", href='/page-5', className="mr-1"),
                dbc.Button("Home", color="primary", href='/', className="mr-1", style={'margin-left': '20px'}),
            ], style={'textAlign': 'center', 'margin': '20px'}), justify='center'
        ), fluid=True
    ),
])

# html.Div([
#     dbc.Button("Generate prediction", id="prediction-button", className="mr-1", style={'margin': '20px'}),
#     dash_table.DataTable(
#         id='prediction-table',
#         columns=[{"name": "Narratives", "id": "Narratives"}, {"name": "Prediction", "id": "Prediction"}, {"name": "Probabiliy", "id": "Probabiliy"}],
#     ),
# ])



# Updating the page content based on the URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return homepage
    elif pathname == '/page-1':
        return page_1
    elif pathname == '/page-2':
        return page_2
    elif pathname == '/page-3':
        return page_3
    elif pathname == '/page-4':
        return page_4_layout
    elif pathname == '/page-5':
        return page_5_layout
    elif pathname == '/page-6':
        return page_6_layout
#     elif pathname == '/page-6':
#         return page_6_layout
    else:
        return dbc.Jumbotron([
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised...")
        ])

# Callback to toggle sidebar
@app.callback(
    Output("collapse", "is_open"),
    [Input("btn_sidebar", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_sidebar(n, is_open):
    if n:
        return not is_open
    return is_open

# Callbacks for page 1
@app.callback(
    Output('subproduct-dropdown', 'options'),
    Input('product-dropdown', 'value')
)
def update_subproduct_dropdown(selected_product):
    subproducts = product_to_subproduct[selected_product]
    return [{'label': subproduct, 'value': subproduct} for subproduct in subproducts]

@app.callback(
    Output('choropleth-graph', 'figure'),
    Input('product-dropdown', 'value'),
    Input('subproduct-dropdown', 'value')
)
def update_graph(selected_product, selected_subproduct):
    df_subproduct = product_subproduct_state_df[(product_subproduct_state_df['product'] == selected_product) & (product_subproduct_state_df['subproduct'] == selected_subproduct)]
    fig1b = go.Figure(data=go.Choropleth(
        locations=df_subproduct['state'],
        z=df_subproduct['Counts_Complaints'],
        locationmode='USA-states',
        colorscale="Magma_r",
        name=f"{selected_product}: {selected_subproduct}"
    ))
    fig1b.update_layout(
        geo_scope='usa')
    return fig1b

# Callbacks for page 2
@app.callback(
    Output('bar-graph', 'figure'),
    Input('issue-dropdown', 'value')
)
def update_graph(selected_issue):
    df_issue = issue_subissue_df[(issue_subissue_df['issue'] == selected_issue)]
    fig2a = px.bar(df_issue, x='subissue', y='count', labels={'subissue':'Sub-Issues', 'count':'Count'})
    fig2a.update_layout(title_text=f"Distribution of sub-issues for {selected_issue}")
    return fig2a

# Callbacks for page 3
@app.callback(
    Output('chi-square-test', 'children'),
    Input('url', 'pathname'),
    methods=['GET']  # Add this line to specify the allowed methods
)
def perform_chi_square_test(pathname):
    if pathname == '/page-3':
        # Create a cross-tabulation (contingency table)
        contingency_table = pd.crosstab(method_df_pd['submitted_via'], method_df_pd['company_response_to_consumer'])

        # Perform the Chi-Square Test of Independence
        chi2, p, dof, expected = chi2_contingency(contingency_table)

#         # Create a Dash DataTable from the contingency table
#         table = dash_table.DataTable(
#             data=contingency_table.reset_index().to_dict('records'),
#             columns=[{'name': i, 'id': i} for i in contingency_table.reset_index().columns]
#         )

        # Test result interpretation
        if p < 0.05:
            interpretation = "The p-value is less than 0.05, we reject the null hypothesis and conclude that there is a significant relationship between the submission method and the company response."
        else:
            interpretation = "The p-value is greater than 0.05, we fail to reject the null hypothesis. This suggests that there's no statistically significant relationship between the method used to submit the complaint (submitted_via) and the response from the company (company_response_to_consumer)."

        # Return formatted string
        return [
            #html.H3("Contingency Table"),
            #table,  # Display the contingency table
            html.Br(),
#             html.H3("Chi-Square Test Result"),
#             html.P(f"Chi2 Statistic: {chi2}, P-value: {p}"),
            html.Div(className="card text-white bg-primary mb-3", style={"max-width": "80rem"}, children=[
            html.Div(className="card-header", children="Chi-Square Test Results"),
            html.Div(className="card-body", children=[
            #html.H4(className="card-title", children="Chi-Square Test Results"),
            html.P(className="card-text", children=f"Chi-square statistic: {chi2}, p-value: {p}, Degrees of freedom: {dof}"),
            html.P("Interpretation: " + interpretation)# Replace X, Y,  with the actual results
        ]),
    ])
        ]
    else:
        return ""

# # Callbacks for page 4
@app.callback(
    Output('graph_4a', 'figure'),
    Input('year-slider', 'value')
)
def update_graph(selected_year):
    filtered_df = complaints_over_time[complaints_over_time['date_received'].dt.year == selected_year]

    return {
        'data': [
            go.Scatter(x=filtered_df['date_received'], y=filtered_df['count'],
            mode='lines',
            name='Actual Frequency of Complaints',
            line=dict(color='lightgrey', width=0.5)),

            go.Scatter(x=filtered_df['date_received'], y=filtered_df['monthly_mva'],
                       mode='lines', name='Monthly Moving Average of Complaints')
        ],
        'layout': go.Layout(title='Trend in complaints received over time for the year ' + str(selected_year))
    }

@app.callback(
    Output(component_id='graph_4c', component_property='figure'),
    [Input(component_id='dropdown_4c', component_property='value')]
)
def update_graph(selected_product):
    fig4c = go.Figure()
    fig4c.add_trace(go.Scatter(x=pivot_df_mva.index, y=pivot_df_mva[selected_product], mode='lines', name=selected_product))
    fig4c.update_layout(
        title=f"Trend in Complaints for {selected_product} Over Time (Moving Average)",
        xaxis_title="Date Received",
        yaxis_title="Number of Complaints (Moving Average)",
        legend_title="Product",
    )
    return fig4c

# # # Callbacks for page 6
@app.callback(
    [
        Output('narrative', 'children'),
        Output('prediction', 'children'),
        Output('prob_max', 'children')
    ],
    [Input('prediction-button', 'n_clicks')],
)
def generate_prediction(n_clicks):
    if n_clicks is None:  # if the button was never clicked
        # Return some default values
        return "No Narrative yet", "No Prediction yet", "No Probability yet"
    else:  # only generate prediction when button is clicked
        Responce_dict = {}
        # Picking a random number
        random_number=random.randint(0, Narrative_test_set.shape[0])
        # Selecting one Narrative
        narrative=Narrative_test_set.iloc[random_number, 0]
        text_to_list = [narrative]
        Responce_dict["Narratives"] = narrative
        # Predicting the reponce
        prediction = Complement_NB.predict(text_to_list)
        prediction = prediction[0]
        Responce_dict["Prediction"] = prediction
        # Storing the probability of the reponce
        prob = Complement_NB.predict_proba(text_to_list)
        prob_max = max(prob[0])*100
        prob_max = f"{prob_max:.2f}"
        Responce_dict["Probabiliy"] = prob_max
        # return data in format for DataTable
        return narrative, prediction, prob_max

# App layout
# app.layout = dbc.Container(
#     [
#         dcc.Location(id='url', refresh=False),
#         dbc.Row(
#             [
#                 dbc.Col([sidebar], style={'padding': 0}, md=2),  # remove padding from sidebar column
#                 dbc.Col(html.Div(id='page-content', style=heading_style), md=9)  # content will take up the rest 9 columns
#             ],
#             style={'margin': 0, 'padding': 0}  # remove margins and padding from the row
#         )
#     ],
#     style={'padding': 0},  # remove padding from the container
#     fluid=True  # set container width to the full available width
# )
app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        navbar,  # Include navbar at the top
        dbc.Row(
            [
                dbc.Col(sidebar_collapse, width=2),  # Sidebar takes 2/12 width
                dbc.Col(
                    [
                        html.Div(id='page-content', style=heading_style),  # content will take up the rest 10 columns
                        footer,  # include footer at the bottom
                    ],
                    md=10
                ),
            ],
            style={'margin': 0, 'padding': 0}  # remove margins and padding from the row
        ),
    ],
    id='main-div'
)



if __name__ == '__main__':
    app.run_server(debug=True)
#Launch http://127.0.0.1:8050 on your browser
#calling app.run_server(), start the ngrok server
# url = ngrok.connect(8050).public_url
# print('Running on:', url)

# if __name__ == '__main__':
#     app.run_server(port=8050)
