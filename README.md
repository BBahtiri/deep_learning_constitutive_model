# A deep-learning based and thermodynamically consistent approach for constitutive modeling of fiber reinforced nanoparticle-filled epoxy nanocomposites under ambient conditions

In this work, we propose a physics-informed deep learning (DL)-based constitutive model for
investigating epoxy based composites under different ambient conditions. The deep-learning
model is trained to enforce thermodynamics principles and ensures a thermodynamic consistent
constitutive model. For this, a long short-term memory network is combined with a
feed-forward neural network to predict internal variables of the material system, which are
needed to characterize the internal dissipation of the material. Another feed-forward neural
network is employed to predict the free-energy function, therefore defining the thermodynamic
state of the whole system. 

The data is directly generated from cyclic loading-unloading experiments conducted on a nanoparticle filled
epoxy system. The experiments include diverse ambient conditions e.g. temperature,
moisture and nanoparticle volume fraction. Accordingly, the accuracy of the deep-learning
model in accurately predicting the material behavior for a material system characterized by a
highly nonlinear response with temperature- and moisture dependency is shown. Importantly,
the deep-learning model solely utilizes experimental data, demonstrating the capability to
capture the complex stress-strain response of the material at hand.

![This is an image](/pinn.PNG)
