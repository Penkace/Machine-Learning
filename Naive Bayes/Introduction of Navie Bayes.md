# Navie Bayes 
### Advantages And Disadvantages
* Advantages: In the case of less data, Navie Bayes is still effective and it can handle multi-classification problems.
* Disadvantages: It's sensitive for input data.
<br>
### The Principle of Naive Bayes
* Conditional Probability: P(A|B) = P(AB)/P(B)
                           P(AB) = P(A|B)*P(B) = P(B|A)*P(A)
                           
* Total Probability Theorem: P(A) = P(A|B_1)*P(B_1) + P(A|B_2)*P(B_2) +...+ P(A|B_n)*P(B_n) 
                             or P(A) = P(A*B_1) + P(A*B_2) +...+ P(A*B_n)
                             
* BayesRule: P(A|B) = P(B|A)*P(A)/P(B)
             P(B) = P(B*A_1) + P(B*A_2) +...+ P(B*A_n)
             (we always call P(A|B) the posterior probability and call P(A) the prior probability.
             In NavieBayes, P(A|B) is the maximum posterior probability)
       
* Naive Bayes Basic Formula:
             y = f(x) = arg max_(ck) P(Y = c_k)
             
<br>
### Why we call Naive Bayes? What's meaning of the word "Naive"?
---
Actually, there is a certain relationship among features. However, in Naive Bayes, we assume features are indepedent of each others, which
will cause a certain error. Scientists tried their best to optimize the algorithm so that the error is close to zero. A posteriori probability 
maximization criterion based on the expected risk minimum criterion(from Li Hang).


### The Example
---
