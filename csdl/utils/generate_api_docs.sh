echo "Generating API Docs"

# user api
mkdir -p ../../docs/docs/lang_ref

echo "---" > ../../docs/docs/lang_ref/model.mdx
echo "title: Model" >> ../../docs/docs/lang_ref/model.mdx
echo "sidebar_position: 1" >> ../../docs/docs/lang_ref/model.mdx
echo "---" >> ../../docs/docs/lang_ref/model.mdx
echo "" >> ../../docs/docs/lang_ref/model.mdx
echo "------------------------------------------------------------------------" >> ../../docs/docs/lang_ref/model.mdx
echo "" >> ../../docs/docs/lang_ref/model.mdx
pydoc-markdown -m csdl.core.model user.yml >> ../../docs/docs/lang_ref/model.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/model.mdx

echo "---" > ../../docs/docs/lang_ref/output.mdx
echo "title: Output" >> ../../docs/docs/lang_ref/output.mdx
echo "sidebar_position: 2" >> ../../docs/docs/lang_ref/output.mdx
echo "------------------------------------------" >> ../../docs/docs/lang_ref/output.mdx
echo "" >> ../../docs/docs/lang_ref/output.mdx
pydoc-markdown -m csdl.core.concatenation user.yml >> ../../docs/docs/lang_ref/output.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/output.mdx

echo "---" > ../../docs/docs/lang_ref/simulator_base.mdx
echo "title: SimulatorBase" >> ../../docs/docs/lang_ref/simulator_base.mdx
echo "sidebar_position: 2" >> ../../docs/docs/lang_ref/simulator_base.mdx
echo "---" >> ../../docs/docs/lang_ref/simulator_base.mdx
echo "" >> ../../docs/docs/lang_ref/simulator_base.mdx
echo "------------------------------------------------------------------------" >> ../../docs/docs/lang_ref/simulator_base.mdx
echo "" >> ../../docs/docs/lang_ref/simulator_base.mdx
pydoc-markdown -m csdl.core.simulator_base user.yml >> ../../docs/docs/lang_ref/simulator_base.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/simulator_base.mdx

echo "---" > ../../docs/docs/lang_ref/custom.mdx
echo "title: Custom Operations" >> ../../docs/docs/lang_ref/custom.mdx
echo "sidebar_position: 3" >> ../../docs/docs/lang_ref/custom.mdx
echo "---" >> ../../docs/docs/lang_ref/custom.mdx
echo "" >> ../../docs/docs/lang_ref/custom.mdx
echo "------------------------------------------------------------------------" >> ../../docs/docs/lang_ref/custom.mdx
echo "" >> ../../docs/docs/lang_ref/custom.mdx
pydoc-markdown -m csdl.core.custom_operation user.yml >> ../../docs/docs/lang_ref/custom.mdx
echo "" >> ../../docs/docs/lang_ref/custom.mdx
pydoc-markdown -m csdl.core.custom_explicit_operation user.yml >> ../../docs/docs/lang_ref/custom.mdx
echo "" >> ../../docs/docs/lang_ref/custom.mdx
pydoc-markdown -m csdl.core.custom_implicit_operation user.yml >>../../docs/docs/lang_ref/custom.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/lang_ref/custom.mdx

# developer api
echo "---" > ../../docs/docs/developer/api.mdx
echo "title: Developer API" >> ../../docs/docs/developer/api.mdx
echo "---" >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
echo "------------------------------------------------------------------------" >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.model dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.simulator_base dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.custom_explicit_operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.custom_implicit_operation dev.yml >>../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.node dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.variable dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.output dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.concatenation dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.implicit_output dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.standard_operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.custom_operation dev.yml >> ../../docs/docs/developer/api.mdx
echo "" >> ../../docs/docs/developer/api.mdx
pydoc-markdown -m csdl.core.subgraph dev.yml >> ../../docs/docs/developer/api.mdx
sed -i -e 's/#### /### /g' ../../docs/docs/developer/api.mdx
