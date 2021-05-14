Add `from csdl_om import Simulator`

`from omtools.api` -> `from csdl`
`Group` -> `Model`
`Problem` -> `Simulator`
`import omtools.api as ot` -> `import csdl` ?
`ot.` -> `csdl.` ?
`declare_variable` -> `declare_variable`
`create_input` -> `create_input`

Model constructors must include `impl=Simulator`

## Examples

### Invalid

```py
prob = Problem()
prob.model = ExampleMultipleMatrixAlong0()
prob.setup(force_alloc_complex=True)
prob.run_model()
```

becomes this

```py
m = ExampleMultipleMatrixAlong0(impl=Simulator)
sim = Simulator(m)
sim.run()
```

### Valid

This

```py
prob = Problem()
prob.model = ExampleMultipleMatrixAlong0()
prob.setup(force_alloc_complex=True)
prob.run_model()
print('M1', prob['M1'].shape)
print(prob['M1'])
```

becomes this

```py
m = ExampleMultipleMatrixAlong0(impl=Simulator)
sim = Simulator(m)
sim.run()
print('M1', sim['M1'].shape)
print(sim['M1'])
```
