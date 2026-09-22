# TODO

## inference

1. Send the context together with the image.
2. Support working with links.
3. Split triggers into separate skills.
4. Simplify the prompt by splitting the main prompt into skills, with a detailed description of what belongs in `signal_category`.
5. Add specific lists of signals.
6. Come up with a solution for RAG (mandatory).
7. Run the neural network in two passes: the first describes the image, the second analyzes it. Evaluate and estimate the computational cost.

## vision_app

1. Add a queue.
2. Add the ability to analyze batches.
3. Add a column for signals to the database.
4. Define windows of a specific size.
5. Display an image thumbnail instead of the filename.
6. Update the functionality for links.
7. Add a chat with the neural network that can search the results based on a specific query.
8. Reconsider the roles.
9. Add `.env` and store some constants there.

### Optional

- Add a light theme.
- Add the ability to upload a profile picture.
