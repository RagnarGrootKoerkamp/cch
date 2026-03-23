# Customizable Contraction Hierarchies

This repo implements CCHs. See the corresponding blogpost for details and
implementation notes:

https://curiouscoding.nl/posts/cch/

In the end it's up to 2x faster than the C++ baseline by doing of lot of SIMD
squeezing, but I'm also skipping the traceback part, so it seems not too
promising for the added complexity in the end.

Current status: abandoned
