# AI-Powered Math Riddle Generator

This project fine-tunes GPT-2 to generate mathematical riddles. 

## Features
- Generate unique math riddles.
- Simple and interactive UI with Streamlit.
- User-friendly input for custom prompts.

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/MaAn-41/Math_Riddle_Generator.git
   cd math-riddles
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the app:
   ```sh
   streamlit run app.py
   ```

## Usage
- Enter a custom riddle prompt.
- Click on "Generate Riddle" to get an AI-generated math riddle.

## Sample Riddles

The dataset includes 30 riddles of varying complexity, including:

- Number puzzles
- Digit manipulation problems
- Algebra riddles
- Fraction problems
- Consecutive number challenges

## Customization

To customize the generation process, you can modify:

- Temperature (0.8 by default) - Higher values make output more random
- Top-k and Top-p values - Controls diversity of generated text
- Max length - Controls maximum length of generated riddles

## Model Details
This project uses a fine-tuned GPT-2 model to generate riddles based on a dataset of mathematical puzzles.

## Contributing
Feel free to open issues or submit pull requests to improve this project.

## License
This project is licensed under the MIT License.
