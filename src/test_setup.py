from config import PROJECT_NAME
from model import create_model


print(f"Project: {PROJECT_NAME}")
print(f"Model: {type(create_model()).__name__}")
print("OceanEmbed 2.0 setup is working!")