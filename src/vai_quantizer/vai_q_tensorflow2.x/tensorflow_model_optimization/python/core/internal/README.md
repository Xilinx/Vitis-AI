# Internal

The internal package contains a collection of modules that are documented,
fully working, and there exists a plan to be graduated to the main `tf.mot`
namespace -- either a literal swap of `tf.mot.internal.<module>` for
`tf.mot.<module>`, or creation of separate public API that calls the internal
API under the hood.

The `tf.mot.internal` modules can change in backwards incompatible manner. Thus,
depending on these modules is not recommended.
