# user api
echo "---\ntitle: Model\nsidebar_position: 1\n---\n\n------------------------------------------------------------------------\n\n" > ../../docs/docs/lang_ref/model.mdx
pydoc-markdown -m csdl.core.model user.yml >> ../../docs/docs/lang_ref/model.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/model.mdx

echo "---\ntitle: Output\nsidebar_position: 2\n---\n\n------------------------------------------------------------------------\n\n" > ../../docs/docs/lang_ref/output.mdx
pydoc-markdown -m csdl.core.explicit_output user.yml >> ../../docs/docs/lang_ref/output.mdx
echo "\n\n" >> ../../docs/docs/lang_ref/custom.mdx
# pydoc-markdown -m csdl.core.implicit_output user.yml >> ../../docs/docs/lang_ref/output.mdx
# sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/implicit_output.mdx

echo "---\ntitle: SimulatorBase\nsidebar_position: 2\n---\n\n------------------------------------------------------------------------\n\n" > ../../docs/docs/lang_ref/simulator_base.mdx
pydoc-markdown -m csdl.core.simulator_base user.yml >> ../../docs/docs/lang_ref/simulator_base.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/simulator_base.mdx

echo "---\ntitle: Custom Operations\nsidebar_position: 3\n---\n\n------------------------------------------------------------------------\n\n" > ../../docs/docs/lang_ref/custom.mdx
pydoc-markdown -m csdl.core.custom_operation user.yml >> ../../docs/docs/lang_ref/custom.mdx
echo "\n\n" >> ../../docs/docs/lang_ref/custom.mdx
pydoc-markdown -m csdl.core.explicit_operation user.yml >> ../../docs/docs/lang_ref/custom.mdx
echo "\n\n" >> ../../docs/docs/lang_ref/custom.mdx
pydoc-markdown -m csdl.core.implicit_operation user.yml >>../../docs/docs/lang_ref/custom.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/custom.mdx

# developer api
echo "---\ntitle: Developer API\n---\n\n------------------------------------------------------------------------\n\n" > ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.model dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.simulator_base dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.explicit_operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.implicit_operation dev.yml >>../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.node dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.variable dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.output dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.explicit_output dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
# pydoc-markdown -m csdl.core.implicit_output dev.yml >> ../../docs/docs/developer/api.mdx
# echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.standard_operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.custom_operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "\n\n" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.subgraph dev.yml >> ../../docs/docs/developer/api.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/developer/api.mdx
