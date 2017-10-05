# nips-adversarial-2017

Codes for the NIPS 2017 adversarial example (AE) competitions:

1.  [Non-targeted AE](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)
2.  [Targeted AE](https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack)
3.  [AE Defense](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack)

This software represents joint work by Mike Pekala, Neil Fendley and I-Jeng Wang.


## Quick Start

See the Makefile for examples of how to run things.  Note that this repo houses the code for all three contests (attack, targeted-attack, and defense).

Before running, you will have to download the necessary checkpoint files and place them in the *Weights* directory.  You will also have to convert the namespace for the adversarially trained inceptionV3 weights (which was provided by competition organizers) so that it does not clash with the namespace of "vanilla" inceptionV3.  See *tf_rename_vars.py*.


## Brief Discussion

### Attacks
These consist of a straightforward ensemble attack against the three networks known to be included as baselines (and potentially as part of some defenses): inceptionV3 and the two adversarially trained networks provided by the competition organizers.  The untargeted attack is just a targeted attack against randomly selected labels (which empirically seemed to work better than our untargeted loss function when the target is an ensemble).  This is not a particularly sophisticated or interesting approach; however, it was used because 

1. the stringent time requirement precluded more computationally expensive methods and 
2. we were very short on development/calendar time at the end due to some issues with our original (keras-based) approach.

There is some strategy involved in selecting which networks to put in the ensemble and how to weight them; more models (or attacking models equipped with additional capabilities, such as Gaussian noise) slow down convergence to a solution.  Ultimately, the uncertainty in evaluation platform runtime and the "no box" nature of the defenses led to employ this very conservative approach.    With more runtime I would have at least included some of the noisy models in the attack (which, based on limited experiments, do indeed seem to make progress against this type of defense given enough GPU time), and also additional networks.  All of these can be done with the code included here.


### Defense
On the defense side, we use a simple two model ensemble with additive Gaussian noise; again, this is neither novel nor particularly interesting.  This was not the approach we originally envisioned (or implemented); however, the aforementioned time constraints led us to set aside our more ambitious program for something modest that should (hopefully) run in the time required and work well enough against attacks that do not generalize well.


## Disclaimers

It is not clear as of this writing that our codes will run successfully on the remote (evaluation) system; we unfortunately did not get any successful feedback until the third development round so this submission is a bit of a flier.

Note also the software and comments do not reflect our usual standards - we were in a bit of a hurry at the end.

The code in this repository is a copy of code from an internal (private) repository we were using during the competition. The latest version of all files should be here; however, if you find anything missing or out-of-date please contact the author.

<sub>
THE SOFTWARE AND ITS DOCUMENTATION ARE PROVIDED AS IS AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES WHATSOEVER. ALL WARRANTIES INCLUDING, BUT NOT LIMITED TO, PERFORMANCE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT ARE HEREBY DISCLAIMED. USERS ASSUME THE ENTIRE RISK AND LIABILITY OF USING THE SOFTWARE. USERS ARE ADVISED TO TEST THE SOFTWARE THOROUGHLY BEFORE RELYING ON IT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DAMAGES WHATSOEVER, INCLUDING, WITHOUT LIMITATION, ANY LOST PROFITS, LOST SAVINGS OR OTHER INCIDENTAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF THE USE OR INABILITY TO USE THE SOFTWARE.
</sub>
