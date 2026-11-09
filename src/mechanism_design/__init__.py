"""Monte Carlo study of auction mechanisms and their incentive properties.

Three mechanisms are studied, each probing the same question from a different
angle: when does a mechanism make honest reporting the best available strategy?

* ``second_price`` -- the Vickrey auction, where truthfulness is weakly dominant.
* ``first_price``  -- where truthfulness is *not* an equilibrium and bidders shade.
* ``vcg_path``     -- VCG applied to buying the cheapest path in a network.
"""

__version__ = "1.0.0"
