Thinking about different reward functions

1. Combined Count: This just counts the number of squares combined. It doesn't necessarily train a model to combine the largest
numbers, but rather to live as long as possible. It also teaches the model that 64 + 64 is equivalent to 2 + 2, which is not right.
So it's not very good. 

2. Highest number on board: 

3. Total tiles: The sum of tiles on the game board. 

3. Combined Numbers: This is where I give it a reward of 2 if it accomplishes 2 + 2, 64 if it accomplishes 64 + 64. It's like 
Combined Count but it actually incentivizes the model to combine larger numbers.

4. Survival: Turns survived (or, number of moves made). I'd imagine this is correlated heavily wtih combined count. But it might be a useful
second order reward function that we can combined with something like Combined Numbers.