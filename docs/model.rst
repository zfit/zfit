Building a model
================

In order to build a generic model the concept of function and distributed density functions (PDFs) need to be clarified. 
The PDF, or density of a continuous random variable, of X is a function f(x) that describes the relative likelihood for this random variable to take on a given value. In this sense, for any two numbers a and b with $$a \leq b $$, 

$$P(a \leq X \leq b) = \int^{b}_{a}f(X)dx$$

That is, the probability that X takes on a value in the interval [a, b] is the area above this interval and under the graph of the density function.
In other words, in order to a function to be a PDF it must satisfy two criteria: (1) $$f(x) \neq 0$$ for all x; (2) $$int^{\infty}_{-\infty}f(x)dx = $$ are under the entire graph of $$f(x)=1$$. 
In ``zfit`` these distinctions are respect, i.e.  conditions in whcih  

Extended PDF
============

Composite PDF
=============

Custom PDF
==========


Sampling from a PDF
===================





