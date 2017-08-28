# rnn-google-stock-prediction

Recurrent Neural Networks
- venturing into complex, forward-looking and cutting-edge of Deep Learning

Outline: 
- The idea behind Recurrent Neural Networks, compare them to the human brain and what makes them special in comparison to regular ANN.
- The Vanishing Gradient Problem - used to prevent them to moving forward
- Long Short-Term Memory (LSTM) - architecture, very exciting, complex structure
- Practical Intuition - examples of researchers, intuitive, how they'll think (neuroscience)
- Extra -> LTSM Variations

- Supervised Deep Learning 
  - Artificial Neural Networks (ANN): Used for Regression and Classification
  - Convolutional Neural Networks (CNN): Used for Computer Vision
  - Recurrent Neural Networks (RNN): Used for Time Series Anaylsis
- Unsupervised Deep Learning
  - Self-Organizing Maps: Used for Feature Detection
  - Deep Boltzmann Machines: Used for Recommendation Systems
  - AutoEncoders: Used for Recommendation Systems

- Idea behind Recurrent Neural Networks (RNN)
  - RNN is one of the most advanced algorithms for Supervised Deep Learning
  - Human Brain - where we're in the map of the brain
    - Deep Learning is to mimick the human brain and get similar functions and leverage what evolution already has for us
    - Brain 
      - Cerebrum 
        - Frontal Lobe
          - Recurrent Neural Networks
          - Frontal lobe responsible for short-term memory, personality, motor cortex, behavior, working memory
        - Temporal Lobe
          - Artificial Neural Networks
          - Temporal responsible for long-term memory
          - weights represent long-term memory
          - Recognition memory (long-term)
        - Parietal Lobe
          - Sensation and perception and creating a spatial coordination system represent the world around them
        - Occipital Lobe
          - Convolutional Neural Networks (CNN)
      - Cerebellum (Latin for little brain)
      - Brainstem - connects brain to organ
      - ANN main advantage? breakthrough, aside from backpropagation (apply everything)
        - the weights, can learn through prior experience (observation, epochs)
        - weights are present in all neurons in the brain
        - weights represent long-term memory, process the same tomorrow, etc.
   - RNN is like short-term memory, they can remember things that just happened in the previous couple of observations and apply that knowledge going forward
   - Neural Networks
    - Input Layer, Hidden Layer, Output Layer
    - How to change to RNN? Squash everything, think of it looking underneath neural network, new dimension, flatten out, neurons still there, twist the entire thing and make it vertical
    - Hidden layer - temporal loop, not only gives an output but feeds itself, common approach to unwind loop, whole layer of neurons
    - Neurons connecting to each through time, previous neurons
    - Allows them to pass information to themselves in the future and analyze them
  - Examples
    - One-to-Many Relationship
      - One input and many outputs
      - Image where computer describes an image
        - CNN -> RNN, come up with description
        - Long-term memory, feature recognition system, RNN makes sense out of the sentence ('black and white dog jumps over bar... nouns and verbs)
    - Many to One 
      - Sentiment anaylsis 
        - Lots of text, is it positive comment or negative comment 
    - Many to Many
      - Google Translator 
      - Gender (languages, one replacement, changes the sentence, depend on the words of the changed, short-term of the previous word to know the next)
      - Subtitle movies
      - RNN movie, LTSM written by Benjamin - https://www.youtube.com/watch?v=LY7x2Ihqjmc (9 Minutes)
      - https://arstechnica.com/gaming/2016/06/an-ai-wrote-this-movie-and-its-strangely-moving/
      - Construct sentences for the most parts, lacks a bigger picture, seperate sentences (90% makes sense) but linking sentences together

- Vanishing Gradient Problem
  - Cost Function, Gradient Descent C = 1/2 (y^ - y)^2
  - Win, Wrec, Wout
  - RNN's Node is a representation of a whole layer
  - Cost function compares what you should be getting
  - Wrec = weight recurring, multiplying the weights multiple time
  - Problem arises: multiplying small, it gets smaller, weights assigned close to zero; if Wrec starts close to zero, gradient becomes even less
  - Vanishing gradident is bad the lower the gradient the harder the network can update the weights
  - The lower the gradient is slower, the higher the gradient the faster it can update the weights
  - viscious cycle, training is slow, training on the disbalance, the whole network is not being trained correctly because of the weights. Domino effect
  - Wrec ~ small => Vanishing
  - Wrec ~ large => Exploding
  - Solutions:
    1. Exploding Gradient
      - Truncated Backpropagation
      - Penalties 
      - Gradient Clipping (maximum value)
    2. Vanishing Gradient
      - Weight initialization 
      - Echo State Networks
      - Long Short-Term Memory Networks (LSTMs)
  - Additional Reading:
    - Sepp (Josef) Hochreiter, 1991, Untersuchungen zu dynamischen neuronalen Netzen
    - Yoshua Bengio, 1994, Learning Long-Term Dependencies with Gradient Descent is Difficult
    - Recommend: Razvan Pascanu, 2013, On the difficulty of training recurrent neural networks - http://proceedings.mlr.press/v28/pascanu13.pdf
