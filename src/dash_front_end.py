from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# Custom theme colors
custom_theme = {
    'primary': '#0A2647',  # Deep Blue
    'secondary': '#144272',  # Medium Blue
    'accent': '#205295',  # Light Blue
    'highlight': '#2C74B3',  # Bright Blue
    'light': '#FFFFFF',  # White
    'gray': '#F5F5F5',  # Light Gray
    'text': '#333333',  # Dark Gray
    'success': '#28A745',  # Green
    'warning': '#FFC107',  # Yellow
    'danger': '#DC3545'  # Red
}

# Custom CSS
custom_css = f"""
:root {{
    --primary: {custom_theme['primary']};
    --secondary: {custom_theme['secondary']};
    --accent: {custom_theme['accent']};
    --highlight: {custom_theme['highlight']};
}}
body {{
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background-color: {custom_theme['gray']};
    color: {custom_theme['text']};
}}
.nav-custom {{
    background-color: var(--primary);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.card {{
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border: none;
    border-radius: 8px;
}}
.btn-custom-primary {{
    background-color: var(--accent);
    border: none;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.btn-custom-primary:hover {{
    background-color: var(--highlight);
    transform: translateY(-1px);
    transition: all 0.2s;
}}
.feature-icon {{
    font-size: 2rem;
    color: {custom_theme['accent']};
}}
.section-title {{
    color: {custom_theme['primary']};
    border-bottom: 2px solid {custom_theme['accent']};
    padding-bottom: 10px;
    margin-bottom: 30px;
}}
"""

# Initialize Dash app
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
    ]
)

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
{{%metas%}}
<title>Enterprise ML Platform</title>
{{%favicon%}}
{{%css%}}
<style>
{custom_css}
</style>
</head>
<body>
{{%app_entry%}}
<footer>
{{%config%}}
{{%scripts%}}
{{%renderer%}}
</footer>
</body>
</html>
"""

# Navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="#home")),
        dbc.NavItem(dbc.NavLink("Features", href="#features")),
        dbc.NavItem(dbc.NavLink("About", href="#about")),
        dbc.NavItem(dbc.NavLink("Contact", href="#contact")),
    ],
    brand=html.Span([
        html.I(className="fas fa-brain mr-2"),
        "Enterprise ML Platform"
    ]),
    brand_href="#",
    className="nav-custom mb-4",
    dark=True
)

# Main layout
app.layout = html.Div([

    navbar,

    dbc.Container([

        # Hero Section
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("Enterprise ML Inference Platform", className="display-4 mb-3"),
                    html.P(
                        "Empower your business with state-of-the-art machine learning inference capabilities",
                        className="lead mb-4"
                    ),
                ], className="hero-section text-center")
            ])
        ], className="mb-5"),

        # Main Interface Section
        dbc.Row([
            # Model Configuration Card
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Model Configuration", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.Label("Select Model Version:", className="fw-bold"),
                        dcc.Dropdown(
                            id='model-config-dropdown',
                            options=[
                                {'label': 'Enterprise Small', 'value': 'small'},
                                {'label': 'Enterprise Medium', 'value': 'medium'},
                                {'label': 'Enterprise Large', 'value': 'large'}
                            ],
                            value='medium',
                            className='mb-4'
                        ),
                        html.Label("Upload Data:", className="fw-bold"),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-2x mb-2"),
                                html.Br(),
                                'Drag and Drop or ', html.A('Select File')
                            ]),
                            className='border rounded p-4 text-center mb-4'
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-play mr-2"), "Run Inference"],
                            id='run-inference-button',
                            className='btn-custom-primary w-100'
                        )
                    ])
                ])
            ], md=6),

            # Results Card
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4("Results Dashboard", className="mb-0"),
                    ]),
                    dbc.CardBody([
                        html.Div(id='inference-results')
                    ])
                ])
            ], md=6)
        ], className="mb-5"),

        # Features Section
        html.Div([
            html.H2("Enterprise Features", className="section-title text-center", id="features"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-shield-alt feature-icon mb-3"),
                            html.H4("Enterprise Security"),
                            html.P("Bank-grade security with encryption at rest and in transit.")
                        ], className="text-center")
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-tachometer-alt feature-icon mb-3"),
                            html.H4("High Performance"),
                            html.P("Optimized inference pipeline for maximum throughput.")
                        ], className="text-center")
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.I(className="fas fa-clock feature-icon mb-3"),
                            html.H4("24/7 Support"),
                            html.P("Round-the-clock enterprise support and monitoring.")
                        ], className="text-center")
                    ])
                ], md=4),
            ], className="mb-5")
        ]),

        # About Section
        html.Div([
            html.H2("About Our Platform", className="section-title text-center", id="about"),
            dbc.Row([
                dbc.Col([
                    html.P("""
                        Our Enterprise ML Platform is designed to meet the demanding needs of modern businesses.
                        With a focus on reliability, security, and performance, we provide a robust solution for enterprise-scale machine learning deployment.
                    """, className="lead text-center mb-4"),

                    # Additional Information Cards
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Enterprise Ready"),
                                    html.P("Built for scale with enterprise-grade security and compliance.")
                                ])
                            ])
                        ], md=4),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("Cloud Native"),
                                    html.P("Leveraging cloud infrastructure for optimal performance.")
                                ])
                            ])
                        ], md=4),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("API First"),
                                    html.P("RESTful APIs for seamless integration with your stack.")
                                ])
                            ])
                        ], md=4),
                    ], className="mb-4")
                ])
            ])
        ]),

        # Footer
        html.Footer([
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("© 2024 Enterprise ML Platform. All rights reserved.", className="text-center text-muted")
                ])
            ])
        ])

    ], fluid=True, className="p-4")
])

# Add your callbacks here (same as before)

if __name__ == '__main__':
    app.run(debug=True)