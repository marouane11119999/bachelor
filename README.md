# COVID Data Integrator

**Bachelor Project**

A simple application to integrate and clean COVID-19 and Long COVID-19 data.

## Prerequisites

Ensure you have the following installed on your system:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Installation and Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/your-username/covid-data-integrator.git
   ```

2. **Build the Docker containers**
   ```sh
   docker-compose build
   ```

3. **Start the application**
   ```sh
   docker-compose up -d
   ```

4. **Check the application logs**
   ```sh
   docker-compose logs -f pythonapp
   ```

5. **Access the application**
   Open your browser and go to:
   ```
   http://localhost
   ```

## Stopping the Application

To stop the running containers, execute:
```sh
   docker-compose down
```

