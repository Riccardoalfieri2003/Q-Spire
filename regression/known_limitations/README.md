Cases that the fixed version still gets wrong, kept for reference (not part of the suite).

- r29_nc_coerced_pub.py: NC is reported although the parameters are bound through
  coerced SamplerPub objects. NC runs the script with the mock circuit, so
  SamplerPub.coerce fails and the binding is not seen. Only PUB tuples
  (circuit, values) passed directly to run() are recognized.
