import json
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEVCONTAINER_DIR = PROJECT_ROOT / ".devcontainer"
DEVCONTAINER_JSON_PATH = DEVCONTAINER_DIR / "devcontainer.json"


def manage_service():
    """Adds or removes a service and its container, providing full context to Docker Compose."""
    try:
        _, operation, service_name = sys.argv
    except ValueError:
        print("Usage: python manage_service.py <add|remove> <service_name>")
        sys.exit(1)

    service_file_to_manage = f"compose-services/docker-compose.{service_name}.yml"
    absolute_service_path = DEVCONTAINER_DIR / service_file_to_manage

    if operation == "add" and not absolute_service_path.exists():
        print(
            f"❌ ERROR: Service '{service_name}' is not installed in this project.")
        print("   Please run 'git pull' to get the latest updates and try again.")
        sys.exit(1)

    with open(DEVCONTAINER_JSON_PATH, "r") as f:
        config = json.load(f)

    # This is the list of ALL compose files currently defining the environment
    active_compose_files = config.get("dockerComposeFile", [])
    if not isinstance(active_compose_files, list):
        active_compose_files = [active_compose_files]

    # --- This is the core logic ---
    if operation == "add":
        if service_file_to_manage not in active_compose_files:
            active_compose_files.append(service_file_to_manage)
            print(f"✅  Added {service_name} to the configuration.")
        else:
            print(f"🟡  {service_name} is already in the configuration.")
    elif operation == "remove":
        if service_file_to_manage in active_compose_files:
            print(f"⚡ Stopping and removing the '{service_name}' container...")
            try:
                # --- NEW, CONTEXT-AWARE COMMAND BUILDING ---
                # Build the full docker compose command with a -f flag for each active file
                command = ["docker", "compose"]
                for file in active_compose_files:
                    command.extend(["-f", str(DEVCONTAINER_DIR / file)])

                # Add the 'rm' command and arguments
                command.extend(["rm", "-s", "-f", service_name])

                subprocess.run(command, check=True, capture_output=True)
                print(f"✅  Container for '{service_name}' has been removed.")
                # --- END OF NEW LOGIC ---

                # Now that the container is removed, update the config file
                active_compose_files.remove(service_file_to_manage)
                print(f"🗑️  Removed {service_name} from the configuration.")

            except subprocess.CalledProcessError as e:
                print(
                    f"🟡  Error while removing container: {e.stderr.decode()}")

        else:
            print(f"🟡  {service_name} was not found in the configuration.")

    config["dockerComposeFile"] = active_compose_files
    with open(DEVCONTAINER_JSON_PATH, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print('\n👉 Next Step: Run "Dev Containers: Rebuild Container" from the Command Palette.')


if __name__ == "__main__":
    manage_service()