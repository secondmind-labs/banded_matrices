# TODO

- Some but not all operators have an attribute for their result_lower_bandwidth
  and result_upper_bandwidth. It would be nice to be systematic. We don't want
  to do that in a way that imposes extra arguments on the operator. One way to
  do that is to always add the attributes at registration in the C++ code but
  not expose them in the banded.py function which should instead set them.

  (Related work by @egor https://github.com/Prowler-io/main/pull/397 is proposing
  more radical API changes where the use of all attributes including lower/upper
  bandwidths is much simplified)

- For some operators we expose combinations of transpose/symmetrise Boolean
  flags because they are needed for the gradients of other operators, but we
  don't have the gradient of the operator with these flags set to True. This
  would be more uniform.

- Artem is of the view that the banded_matrix abstraction should be removed
  altogether, and that every operator should be written directly in terms of the
  underlying Eigen dense matrix. This could be used to simplify index
  manipulations (each access needs an additional subtraction), and enable
  additional optimizations based on Eigen's vectorized operations, for instance.
  This can be done one operator at a time, evaluating the perf benefits every
  time.

- We have some inconsistent numerical precision in release and debug...
  See the pragma and comments in inverse.cc.

- Cholesky has a test for Positive definiteness of input which is crude (if
  the resulting L is such that L L^T is not nearly the input we display a
  warning on cout) but crucial - this issue arises often in some models.
  We should think of an automatic way to test this (additional attribute that
  guards the check, but makes it available even in Release?)

NOTES about cache-performance, and performance improvements in general:

The code has many TODO s related to specific possible optimizations.

Note that the TensorFlow layout is row-major by default. Some of our functions
are in contrast written with a column major layout in mind, which is the Eigen default.

During the migration from prototype we had comments from reviewers about how various
nested loops could be restructured to improve cache efficiency. In particular
swapping the order of some nested loops could improve efficiency. This would need
to be confirmed with careful experimentation.

@bedder notes:

- about the block-band operator:
If I'm understanding correctly and matrices are stored in row-major format
(i.e. each row is contiguous in memory), then to be cache friendly
I think we'd want to loop `row_block` -> `sub_block`-> `col_block`
to be as cache-friendly as possible.

- about `check_zeros_out_of_band` in the pack_matrix operator:
Same comment as before - if rows are kept as contiguous in memory,
we would want to swap the order of these loops to be a bit more cache friendly (I think?).

- about product.hpp
As before - I _think_ there'll be a perf increase from swapping these loops.
