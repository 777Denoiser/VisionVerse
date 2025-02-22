#implements the CLI using argparse to take text input and generate images

import argparse
from src.generation.generator import generate_image

def main():
    parser = argparse.ArgumentParser(description="Generate images from text using CLIP and SIREN.")
