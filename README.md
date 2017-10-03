# nips-adversarial-2017

Codes for the NIPS 2017 adversarial example competition.

Note this software represents joint work by Mike Pekala, Neil Fendley and I-Jeng Wang.


## Quick Start

See the Makefile for examples of how to run things.  Note that this repo houses the code for all three contests (attack, targeted-attack, and defense).

Before running, you will have to download the necessary checkpoint files and place them in the *Weights* directory.  You will also have to convert the namespace for the adversarially trained inceptionV3 weights (which was provided by competition organizers) so that it does not clash with the namespace of "vanilla" inceptionV3.  See *tf_rename_vars.py*.


## Brief Discussion

On the attack side, this code is nothing more than a straightforward ensemble attack against the three networks known to be included as baselines (and potentially as part of some defenses).  The untargeted attack is just a targeted attack against randomly selected labels.  This is not a particular sophisticated or interesting approach; however, it was used because (a) of the stringent time requirement which precluded more computationally expensive methods and (b) we were very short on development/calendar time at the end due to some issues with our original (keras-based) approach.
There is some strategy involved in balancing which networks to put in the ensemble; more models (or attacking models equipped with Gaussian noise) slow down convergence to a solution.  Ultimately, the uncertainty in evaluation platform runtime and the "no box" nature of the defense led to this very conservative approach.    With more runtime I would have at least included some of the noisy models in the attack, and probably a few other networks (all of which can be done easily with the existing code).

On the defense side, we use a simple two model ensemble with some additive Gaussian noise.  Again, this was not the approach we originally envisioned or implemented; the aforementioned time constraints led us to set aside our more ambitious program for something modest that should hopefully run in the time required and work well enough against weak attacks.


## Disclaimers

It is not clear as of this writing that our codes will run successfully on the remote (evaluation) system; we unfortunately did not get any successful feedback until the third development round so this submission is a bit of a flier.

Note also the software and comments do not reflect our usual standards - we were in a bit of a hurry at the end.

The code in this repository is a copy of code from an internal (private) repository we were using during the competition. The latest version of all files should be here; however, if you find anything missing or out-of-date please contact the author.

<sub>
THE SOFTWARE AND ITS DOCUMENTATION ARE PROVIDED AS IS AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES WHATSOEVER. ALL WARRANTIES INCLUDING, BUT NOT LIMITED TO, PERFORMANCE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT ARE HEREBY DISCLAIMED. USERS ASSUME THE ENTIRE RISK AND LIABILITY OF USING THE SOFTWARE. USERS ARE ADVISED TO TEST THE SOFTWARE THOROUGHLY BEFORE RELYING ON IT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DAMAGES WHATSOEVER, INCLUDING, WITHOUT LIMITATION, ANY LOST PROFITS, LOST SAVINGS OR OTHER INCIDENTAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF THE USE OR INABILITY TO USE THE SOFTWARE.
</sub>
