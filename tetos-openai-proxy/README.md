```markdown
# OpenAI API Proxy Deployment

This guide will walk you through deploying the **tetos TTS proxy** using Docker. This allows you to run the application in a containerized environment, making it portable and easy to manage.

> **Note:** I take no credit here, the original author did all the work and then Gemini did the heavy lifting for me.

---

## Prerequisites

Before you begin, ensure you have the following installed and configured:

- **Git**
- **Docker Desktop** or **Docker Engine**
- Your cloud credentials (e.g., `google-credentials.json` file, Azure Speech Key & Region) ready.

---

## Step 1: Clone the Repository

First, download the project files from the source repository. Open your terminal and run:
---

## Step 2: Navigate to the Project Directory

Change your current location in the terminal to the directory containing the Dockerfile and the proxy script.

```bash
cd tetos\tetos-openai-proxy
```
---

## Step 3: Build the Docker Image

This command builds the Docker image from the Dockerfile in the current directory. The `-t tetos-proxy` flag tags the image with the name `tetos-proxy`, making it easy to reference later.

```bash
docker buildx build -t tetos-proxy .
```

---

## Step 4: Run the Docker Container

This command starts the container from the image you just built. Let's break down what each flag does:

```bash
docker run -d --rm \
  -p 8787:8888 \
  -v ~/.google-credentials.json:/home/appuser/google-credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS="/home/appuser/google-credentials.json" \
  -e AZURE_SPEECH_KEY="$AZURE_SPEECH_KEY" \
  -e AZURE_SPEECH_REGION="$AZURE_SPEECH_REGION" \
  --name my-tetos-proxy \
  tetos-proxy
```

- `-d`: **Detached Mode**. Runs the container in the background and prints the container ID.
- `--rm`: **Remove**. Automatically removes the container when it stops, which is useful for keeping your system clean.
- `-p 8787:8888`: **Port Mapping**. Forwards traffic from port 8787 on your host machine to port 8888 inside the container.
- `-v ~/.google-credentials.json:/home/appuser/google-credentials.json`: **Volume Mount**. Mounts your local Google credentials file into the container so the application can use it for authentication.
    - **Important:** Make sure the path `~/.google-credentials.json` matches the actual location of your file.
- `-e GOOGLE_APPLICATION_CREDENTIALS=...`: **Environment Variable**. Sets the path for the Google credentials inside the container.
- `-e AZURE_...`: Sets the environment variables for your Azure Speech credentials from your current shell session.
    - **Important:** You must have `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` exported as environment variables in your terminal for this to work.
- `--name my-tetos-proxy`: Assigns a convenient name to your running container.
- `tetos-proxy`: The name of the image to run.

---

Your proxy server is now running and accessible on your machine at [http://localhost:8787](http://localhost:8787).
```
