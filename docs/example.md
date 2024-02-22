# AIFeel Documentation: Example

A monad is just a monoid in the category of endofunctors, what's the problem?

As long as it satisfies:
- Left identity: `return a >>= f` is the same as `f a`
- Right identity: `m >>= return` is the same as `m`
- Associativity: `(m >>= f) >>= g` is the same as `m >>= (\x -> f x >>= g)`
- Functor laws: `fmap f xs` is the same as `xs >>= return . f`

Then it's a monad.