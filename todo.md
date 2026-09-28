# TODO

## inference

1. Send the context together with the image.
2. Support working with links.
3. Split triggers into separate skills.
4. Simplify the prompt by splitting the main prompt into skills, with a detailed description of what belongs in `signal_category`.
5. Add specific lists of signals.
6. Come up with a solution for RAG (mandatory).
7. Run the neural network in two passes: the first describes the image, the second analyzes it. Evaluate and estimate the computational cost.
8. remove the S-n id from json as it have no practical use

## vision_app

1. Define windows of a specific size.
2. Display an image thumbnail instead of the filename.
3. Update the functionality for links.
4. Reconsider the roles.
5. Add `.env` and store some constants there.

### Optional

- Add the ability to upload a profile picture.
