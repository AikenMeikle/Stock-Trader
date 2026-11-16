This was a trading strategy I came up with after looking at the s&P 500 priced in gold, I noticed that it follows a somewhat sinusoidal pattern.
I first tried to use a SciPy smoothing spline to fit this curve and then find a stationary point. At first, I thought this worked but after looking in more detail, I realised it was essentially a coin flip between deciding to invest in SPY and GLD. This is because the gradient of a smoothing spline is noisy at the end points.
Then, I tried a quadratic regression model using NumPy to fit a polynomial curve. Again, I thought this worked at the start but during this part in the project, I decided to explore AI agents for the first time and they are very powerful. They helped me analyse my code and showed to me that most of the gains came from a single shoot of luck in 2015.

To conclude, this strategy does not work but it shows the importance of testing the output instead of solely looking at the anaual returns.
