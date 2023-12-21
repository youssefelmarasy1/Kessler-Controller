# Fuzzy Controller for Kessler Asteroids Game

## Overview
Welcome to the Fuzzy Controller for the Kessler Asteroids Game project! In this repository, we aim to implement a fuzzy logic controller for playing the Kessler implementation of the classic asteroids game. The challenge is to design a controller that surpasses the performance of a basic test controller, which fires continuously without any movement (without either thrust or turn).

## Project Goal
The primary objective of this project is to create an intelligent fuzzy logic controller that not only survives but outlasts the test controller. The fuzzy controller should demonstrate improved decision-making, dynamic movement, and strategic firing to navigate the asteroids and avoid collisions while achieving a higher score.

## Getting Started
To run the project and observe the fuzzy controller in action, follow these steps:

### Clone the repository to your local machine:

`git clone https://github.com/youssefelmarasy1/Kessler-Controller.git`

#### Navigate to the project directory:
`cd Kessler-Controller`

### Open the Kessler Asteroids Game environment with the provided test controller.

### Run the fuzzy controller code (my_controller.py) alongside the test controller (test_controller.py).

## File Structure
my_controller.py: Contains the implementation of the Fuzzy Logic Controller using EasyGA for genetic algorithm optimization.

test_controller.py: Provides the code for the test controller, serving as a benchmark for comparison against the developed fuzzy logic controller.

gasys.py: The EasyGA library for genetic algorithm optimization.

graphics_both.py: Graphics module for visualizing the gameplay, copied from the Kessler game GitHub repository.

scenario_test.py: Scenario module for setting up test scenarios, copied from the Kessler game GitHub repository.

README.md: Project documentation providing an overview, instructions, and information about the file structure.

## Usage
Modify the fuzzy controller implementation in fuzzy_controller.py to enhance its decision-making process and improve its ability to navigate through the game environment. Adjust the fuzzy logic rules, membership functions, and input variables to optimize performance.

## Contributing
Feel free to contribute to the project by suggesting improvements, or enhancements, or by submitting bug reports. Create a pull request with your changes or open an issue for discussion.

Happy gaming! 🚀🕹️
