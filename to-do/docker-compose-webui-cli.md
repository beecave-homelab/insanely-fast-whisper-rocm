# Add CLI-based WebUI Launch to Docker Compose

- [x] Update `docker-compose.yaml` to add an optional (commented) command for launching the WebUI via the new CLI script with debug enabled (`python -m insanely_fast_whisper_api.webui.cli --debug`)
- [x] Add a comment in `docker-compose.yaml` explaining the purpose of the new command and how to customize arguments (host, port, etc)
- [x] Test that the CLI command works as expected in the Docker container
- [ ] Update `project-overview.md` to document the new CLI-based launch option
