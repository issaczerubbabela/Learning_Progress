## Video 3
## Introduction to Neural Network
---
A standard neural network is just a function that tries to fit all data in a column.
![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030170202.png/?raw=true)


`partial` function can be used to ==fix== values of ==coefficients== in a function.

```python
from functools import partial
def quad(x,a,b,c) : return a*x**2 + b*x + c
def mk_quad(a,b,c) : return partial(quad,a,b,c)
//now f can be a function with coefficients fixed.
f = mk_quad(3,2,1)
f(1.5)//10.75
```

random noise addition : 
```python
def noise(x, scale): return np.random.normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)
np.random.seed(42)

x = torch.linspace(-2, 2, steps=20)[:,None]
y = add_noise(f(x), 0.15, 1.5)
```

![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030171445.png/?raw=true)
now we need to find a function that overlays all the data in the graph.

```python
@interact(a=1.1, b=1.1, c=1.1)
def plot_quad(a, b, c):
    plt.scatter(x,y)
    plot_function(mk_quad(a,b,c), ylim=(-3,13))
```

we get a plot with sliders that can be used to modify coefficients a, b, c so that we can use them to try and fit the line with the data. but this is inefficient.  
Therefore, we use a numeric measure to obtain the best fit. we use *mean absolute error* to find this. 

>*mean absolute error*
>it is the mean distance of the points in the curve from the points in the graph.

```python
def mae(preds, acts): return (torch.abs(preds-acts)).mean()
```

![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030172136.png/?raw=true)

now we have the measure of the mean absolute error which we can try to minimize manually to find the best fit. Voila! but this can prove to be tedious for large neural networks with millions of parameters. so we use ==calculus== to minimize the error. 
we use derivatives to find the **gradient** of the **mean absolute error** which then we can use to increase or decrease the parameters to decrease the mean absolute error.
this mean absolute error function is called the **loss function**.

```python
def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)
quad_mae([1.1, 1.1, 1.1]) //fixes a,b,c as 1.1
abc = torch.tensor([1.1,1.1,1.1])//creating a tensor from it
abc.requires_grad_() //telling tensor we need gradients for these
loss = quad_mae(abc)
loss //loss is the mean absolute error we are trying to minimize using GRADIENT DESCENT
loss.backward() //to get PyTorch to calculate gradients.
//the gradients are now stored in an attribute called grad
abc.grad //displays the gradient
```

![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030173626.png/?raw=true)

	since the gradients are negative, we observe that our parameters are a little low. So we try to increase them a bit. 
	this can be done by subtracting the gradient by a small multiplication of the gradient.
```python
with torch.no_grad():
    abc -= abc.grad*0.01
    loss = quad_mae(abc)
    
print(f'loss={loss:.2f}')
```
	//our loss has acutally gone down after this.
>*Learning Rate* :
>The small number by which we multiply is called the learning rate. It is the most important hyper-parameter to set when training a neural network. 

we can use a loop to do a few more iterations of this.[^1]

```python
for i in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.01
    print(f'step={i}; loss={loss:.2f}')
```

our loss keeps going down.
once it reaches the correct answer the loss won't go down anymore and it will start getting up instead.
we have to reduce our learning rate as we train so that our parameter doesn't jump past the correct answer.
this is called *learning rate schedule*.

>The trick is that a neural network is a very expressive function. In fact -- it's [infinitely expressive](https://en.wikipedia.org/wiki/Universal_approximation_theorem). **A neural network can approximate any computable function, given enough parameters.** A "computable function" can cover just about anything you can imagine: understand and translate human speech; paint a picture; diagnose a disease from medical imaging; write an essay; etc...

### But how does neural network approximate any given function?
	1. Matrix multiplication, which is just multiplying things together and then adding them up
	2. The function 𝑚𝑎𝑥(𝑥,0)���(�,0), which simply replaces all negative numbers with zero.

This second function is called the Relu or the *Rectified Linear Function*
which can be implemented like :
```python
def rectified_linear(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.) //np.clip(x,0) is same as max(x,0)
```

![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030180106.png/?raw=true)
this is the *ReLU function*
instead of `torch.clip(y, 0.)` , `F.relu(x)` can be used where `F` refers to the `torch.nn.functional` module.
![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030180339.png/?raw=true)

adding two layers of this function turns it into something like this.
![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030180531.png/?raw=true)

this way if enough of this ReLu functions are added up, you could approximate any function with a single input, to whatever accuracy you like. Adding more ReLu just makes it more accurate.
![function](https://github.com/issaczerubbabela/Learning_Progress/blob/main/FastAI/Introduction%20to%20Neural%20Network/Attachments/Pasted%20image%2020231030181310.png/?raw=true)

all we need is enough data and time to train the amazing model.

>*transfer learning*
>: it is when we start a model with parameters of someone else's model.


[^1]: BTW, you'll see we had to wrap our calculation of the new
parameters in `with torch.no_grad()` . That disables the
calculation of gradients for any operations inside that context
manager. We have to do that, because` abc -= abc.grad\*0.01`
isn't actually part of our quadratic model, so we don't want
derivatives to include that calculation.
