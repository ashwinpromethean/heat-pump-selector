# Heat Pump Selector

A Streamlit application for selecting heat pumps based on capacity, temperature conditions, and refrigerant type.

## Features

- User authentication with admin and viewer roles
- Heat pump selection based on heating capacity and operating conditions
- Support for both Air Source and Water Source heat pumps
- Customizable search weights and exact matching options
- Excel data export functionality

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.streamlit/secrets.toml` with your configuration
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## AWS App Runner Deployment

This application is configured for deployment on AWS App Runner using Docker.

### Prerequisites

1. AWS Account with App Runner access
2. GitHub repository with your code
3. Environment variables configured in App Runner

### Required Environment Variables

Set these in your App Runner service configuration:

- `ADMIN_EMAIL`: Admin user email (default: admin@prometheanenergy.com)
- `ADMIN_PASSWORD`: Admin user password
- `VIEWER_EMAIL`: Viewer user email (default: common@prometheanenergy.com) 
- `VIEWER_PASSWORD`: Viewer user password
- `COOKIE_NAME`: Cookie name for auth (default: modelpicker_auth)
- `COOKIE_KEY`: Secret key for cookie encryption (required for security)
- `COOKIE_EXPIRY_DAYS`: Cookie expiry in days (default: 7)
- `DATA_PATH`: Path to Excel data file (default: data/models.xlsx)

### Deployment Steps

1. Push your code to GitHub
2. Create a new App Runner service in AWS Console
3. Connect to your GitHub repository
4. Use "Use a configuration file" and specify `apprunner.yaml`
5. Set the required environment variables
6. Deploy!

## File Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── apprunner.yaml        # AWS App Runner configuration
├── .dockerignore         # Docker ignore file
├── .gitignore           # Git ignore file
├── data/
│   └── models.xlsx      # Heat pump data
└── .streamlit/
    └── secrets.toml     # Local secrets (not in git)
```

## Security Notes

- The `secrets.toml` file is excluded from git for security
- Use strong passwords and a secure cookie key in production
- Environment variables are used for sensitive configuration in production
