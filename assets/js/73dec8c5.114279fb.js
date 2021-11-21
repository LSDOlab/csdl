"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[4678],{3905:function(e,t,n){n.d(t,{Zo:function(){return l},kt:function(){return m}});var i=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function s(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,i)}return n}function p(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?s(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):s(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,i,a=function(e,t){if(null==e)return{};var n,i,a={},s=Object.keys(e);for(i=0;i<s.length;i++)n=s[i],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(i=0;i<s.length;i++)n=s[i],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var u=i.createContext({}),r=function(e){var t=i.useContext(u),n=t;return e&&(n="function"==typeof e?e(t):p(p({},t),e)),n},l=function(e){var t=r(e.components);return i.createElement(u.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},c=i.forwardRef((function(e,t){var n=e.components,a=e.mdxType,s=e.originalType,u=e.parentName,l=o(e,["components","mdxType","originalType","parentName"]),c=r(n),m=a,_=c["".concat(u,".").concat(m)]||c[m]||d[m]||s;return n?i.createElement(_,p(p({ref:t},l),{},{components:n})):i.createElement(_,p({ref:t},l))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var s=n.length,p=new Array(s);p[0]=c;var o={};for(var u in t)hasOwnProperty.call(t,u)&&(o[u]=t[u]);o.originalType=e,o.mdxType="string"==typeof e?e:a,p[1]=o;for(var r=2;r<s;r++)p[r]=n[r];return i.createElement.apply(null,p)}return i.createElement.apply(null,n)}c.displayName="MDXCreateElement"},6232:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return u},metadata:function(){return r},toc:function(){return l},default:function(){return c}});var i=n(7462),a=n(3366),s=(n(7294),n(3905)),p=["components"],o={title:"Custom Operations",sidebar_position:3},u=void 0,r={unversionedId:"lang_ref/custom",id:"lang_ref/custom",isDocsHomePage:!1,title:"Custom Operations",description:"------------------------------------------------------------------------",source:"@site/docs/lang_ref/custom.mdx",sourceDirName:"lang_ref",slug:"/lang_ref/custom",permalink:"/csdl/docs/lang_ref/custom",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/lang_ref/custom.mdx",tags:[],version:"current",sidebarPosition:3,frontMatter:{title:"Custom Operations",sidebar_position:3},sidebar:"docs",previous:{title:"SimulatorBase",permalink:"/csdl/docs/lang_ref/simulator_base"},next:{title:"Array Expansion and Contraction",permalink:"/csdl/docs/std_lib_ref/Array Operations/expand"}},l=[{value:"CustomOperation Objects",id:"customoperation-objects",children:[{value:"initialize",id:"initialize",children:[]},{value:"define",id:"define",children:[]},{value:"add_input",id:"add_input",children:[]},{value:"add_output",id:"add_output",children:[]},{value:"declare_derivatives",id:"declare_derivatives",children:[]}]},{value:"CustomExplicitOperation Objects",id:"customexplicitoperation-objects",children:[{value:"compute",id:"compute",children:[]},{value:"compute_derivatives",id:"compute_derivatives",children:[]},{value:"compute_jacvec_product",id:"compute_jacvec_product",children:[]}]},{value:"CustomImplicitOperation Objects",id:"customimplicitoperation-objects",children:[{value:"evaluate_residuals",id:"evaluate_residuals",children:[]},{value:"compute_derivatives",id:"compute_derivatives-1",children:[]},{value:"solve_residual_equations",id:"solve_residual_equations",children:[]},{value:"apply_inverse_jacobian",id:"apply_inverse_jacobian",children:[]},{value:"compute_jacvec_product",id:"compute_jacvec_product-1",children:[]}]}],d={toc:l};function c(e){var t=e.components,n=(0,a.Z)(e,p);return(0,s.kt)("wrapper",(0,i.Z)({},d,n,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("hr",null),(0,s.kt)("a",{id:"csdl.core.custom_operation"}),(0,s.kt)("h1",{id:"csdlcorecustom_operation"},"csdl.core.custom","_","operation"),(0,s.kt)("a",{id:"csdl.core.custom_operation.CustomOperation"}),(0,s.kt)("h2",{id:"customoperation-objects"},"CustomOperation Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class CustomOperation(Operation)\n")),(0,s.kt)("a",{id:"csdl.core.custom_operation.CustomOperation.initialize"}),(0,s.kt)("h3",{id:"initialize"},"initialize"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def initialize()\n")),(0,s.kt)("p",null,"User defined method to declare parameter values. Parameters are\ncompile time constants (neither inputs nor outputs to the model)\nand cannot be updated at runtime. Parameters are intended to\nmake a ",(0,s.kt)("inlineCode",{parentName:"p"},"CustomOperation")," subclass definition generic, and therefore\nreusable. The example below shows how a ",(0,s.kt)("inlineCode",{parentName:"p"},"CustomOperation")," subclass\ndefinition uses parameters and how the user can set parameters\nwhen constructing the example ",(0,s.kt)("inlineCode",{parentName:"p"},"CustomOperation")," subclass. Note\nthat the user never instantiates nor inherits directly from the\n",(0,s.kt)("inlineCode",{parentName:"p"},"CustomOperation")," base class."),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Example")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"# in this example, we inherit from ExplicitOperation, but\n# the user can also inherit from ImplicitOperation\nclass Example(ExplicitOperation):\n    def initialize(self):\n        self.parameters.declare('in_name', types=str)\n        self.parameters.declare('out_name', types=str)\n\n    def define(self):\n        # use parameters declared in ``initialize``\n        in_name = self.parameters['in_name']\n        out_name = self.parameters['out_name']\n\n        self.add_input(in_name)\n        self.add_output(out_name)\n        self.declare_derivatives(out_name, in_name)\n\n    # define run time behavior by defining other methods...\n\n# compile using Simulator imported from back end...\nsim = Simulator(\n    Example(\n        in_name='x',\n        out_name='y',\n    ),\n)\n")),(0,s.kt)("a",{id:"csdl.core.custom_operation.CustomOperation.define"}),(0,s.kt)("h3",{id:"define"},"define"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def define()\n")),(0,s.kt)("p",null,"User defined method to define custom operation"),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Example")),(0,s.kt)("p",null,".. code-block:: python"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"def define(self):\n    self.add_input('Cl')\n    self.add_input('Cd')\n    self.add_input('rho')\n    self.add_input('V')\n    self.add_input('S')\n    self.add_output('L')\n    self.add_output('D')\n\n    # declare derivatives of all outputs wrt all inputs\n    self.declare_derivatives('*', '*'))\n")),(0,s.kt)("a",{id:"csdl.core.custom_operation.CustomOperation.add_input"}),(0,s.kt)("h3",{id:"add_input"},"add","_","input"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def add_input(name, val=1.0, shape=(1, ), src_indices=None, flat_src_indices=None, units=None, desc='', tags=None, shape_by_conn=False, copy_shape=None)\n")),(0,s.kt)("p",null,"Add an input to this operation."),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Example")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"class Example(ExplicitOperation):\n    def define(self):\n        self.add_input('Cl')\n        self.add_input('Cd')\n        self.add_input('rho')\n        self.add_input('V')\n        self.add_input('S')\n        self.add_output('L')\n        self.add_output('D')\n\n    # ...\n\nclass Example(ImplicitOperation):\n    def define(self):\n        self.add_input('a', val=1.)\n        self.add_input('b', val=-4.)\n        self.add_input('c', val=3.)\n        self.add_output('x', val=0.)\n\n    # ...\n")),(0,s.kt)("a",{id:"csdl.core.custom_operation.CustomOperation.add_output"}),(0,s.kt)("h3",{id:"add_output"},"add","_","output"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def add_output(name, val=1.0, shape=(1, ), units=None, res_units=None, desc='', lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0, tags=None, shape_by_conn=False, copy_shape=None, distributed=None)\n")),(0,s.kt)("p",null,"Add an output to this operation."),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Example")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"class Example(ExplicitOperation):\n    def define(self):\n        self.add_input('Cl')\n        self.add_input('Cd')\n        self.add_input('rho')\n        self.add_input('V')\n        self.add_input('S')\n        self.add_output('L')\n        self.add_output('D')\n\n    # ...\n\nclass Example(ImplicitOperation):\n    def define(self):\n        self.add_input('a', val=1.)\n        self.add_input('b', val=-4.)\n        self.add_input('c', val=3.)\n        self.add_output('x', val=0.)\n\n    # ...\n")),(0,s.kt)("a",{id:"csdl.core.custom_operation.CustomOperation.declare_derivatives"}),(0,s.kt)("h3",{id:"declare_derivatives"},"declare","_","derivatives"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def declare_derivatives(of, wrt, dependent=True, rows=None, cols=None, val=None, method='exact', step=None, form=None, step_calc=None)\n")),(0,s.kt)("p",null,"Declare partial derivatives of each output with respect to each\ninput (ExplicitOperation) or each residual associated with an output with\nrespect to the input/output (ImplicitOperation)."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"class Example(ExplicitOperation):\n    def define(self):\n        self.add_input('Cl')\n        self.add_input('Cd')\n        self.add_input('rho')\n        self.add_input('V')\n        self.add_input('S')\n        self.add_output('L')\n        self.add_output('D')\n\n        # declare derivatives of all outputs wrt all inputs\n        self.declare_derivatives('*', '*')\n\n    # ...\n\nclass Example(ImplicitOperation):\n    def define(self):\n        self.add_input('a', val=1.)\n        self.add_input('b', val=-4.)\n        self.add_input('c', val=3.)\n        self.add_output('x', val=0.)\n        # declare derivative of residual associated with x\n        # wrt x\n        self.declare_derivatives('x', 'x')\n        # declare derivative of residual associated with x\n        # wrt a, b, c\n        self.declare_derivatives('x', ['a','b','c'])\n\n        self.linear_solver = ScipyKrylov()\n        self.nonlinear_solver = NewtonSolver(solve_subsystems=False)\n\n    # ...\n")),(0,s.kt)("a",{id:"csdl.core.custom_explicit_operation"}),(0,s.kt)("h1",{id:"csdlcorecustom_explicit_operation"},"csdl.core.custom","_","explicit","_","operation"),(0,s.kt)("a",{id:"csdl.core.custom_explicit_operation.CustomExplicitOperation"}),(0,s.kt)("h2",{id:"customexplicitoperation-objects"},"CustomExplicitOperation Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class CustomExplicitOperation(CustomOperation)\n")),(0,s.kt)("a",{id:"csdl.core.custom_explicit_operation.CustomExplicitOperation.compute"}),(0,s.kt)("h3",{id:"compute"},"compute"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def compute(inputs, outputs)\n")),(0,s.kt)("p",null,"Define outputs as an explicit function of the inputs"),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Example")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"def compute(self, inputs, outputs):\n    outputs['L'] = 1/2 * inputs['Cl'] * inputs['rho'] * inputs['V']**2 * inputs['S']\n    outputs['D'] = 1/2 * inputs['Cd'] * inputs['rho'] * inputs['V']**2 * inputs['S']\n")),(0,s.kt)("a",{id:"csdl.core.custom_explicit_operation.CustomExplicitOperation.compute_derivatives"}),(0,s.kt)("h3",{id:"compute_derivatives"},"compute","_","derivatives"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def compute_derivatives(inputs, derivatives)\n")),(0,s.kt)("p",null,"User defined method to compute partial derivatives for this\noperation"),(0,s.kt)("p",null,(0,s.kt)("strong",{parentName:"p"},"Example")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"def compute(self, inputs, outputs):\n    outputs['L'] = 1/2 * inputs['Cl'] * inputs['rho'] * inputs['V']**2 * inputs['S']\n    outputs['D'] = 1/2 * inputs['Cd'] * inputs['rho'] * inputs['V']**2 * inputs['S']\n\ndef compute_derivatives(self, inputs, derivatives):\n    derivatives['L', 'Cl'] = 1/2 * inputs['rho'] * inputs['V']**2 * inputs['S']\n    derivatives['L', 'rho'] = 1/2 * inputs['Cl'] * inputs['V']**2 * inputs['S']\n    derivatives['L', 'V'] = inputs['Cl'] * inputs['rho'] * inputs['V'] * inputs['S']\n    derivatives['L', 'S'] = 1/2 * inputs['Cl'] * inputs['rho'] * inputs['V']**2\n\n    derivatives['D', 'Cd'] = 1/2 * inputs['rho'] * inputs['V']**2 * inputs['S']\n    derivatives['D', 'rho'] = 1/2 * inputs['Cd'] * inputs['V']**2 * inputs['S']\n    derivatives['D', 'V'] = inputs['Cd'] * inputs['rho'] * inputs['V'] * inputs['S']\n    derivatives['D', 'S'] = 1/2 * inputs['Cd'] * inputs['rho'] * inputs['V']**2\n")),(0,s.kt)("a",{id:"csdl.core.custom_explicit_operation.CustomExplicitOperation.compute_jacvec_product"}),(0,s.kt)("h3",{id:"compute_jacvec_product"},"compute","_","jacvec","_","product"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def compute_jacvec_product(inputs, d_inputs, d_outputs, mode)\n")),(0,s.kt)("p",null,"[Optional]"," Implement partial derivatives by computing a\nmatrix-vector product"),(0,s.kt)("p",null,(0,s.kt)("em",{parentName:"p"},"Example")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-py"},"def compute(self, inputs, outputs):\n    outputs['area'] = inputs['length'] * inputs['width']\n\ndef compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):\n    if mode == 'fwd':\n        if 'area' in d_outputs:\n            if 'length' in d_inputs:\n                d_outputs['area'] += inputs['width'] * d_inputs['length']\n            if 'width' in d_inputs:\n                d_outputs['area'] += inputs['length'] * d_inputs['width']\n    elif mode == 'rev':\n        if 'area' in d_outputs:\n            if 'length' in d_inputs:\n                d_inputs['length'] += inputs['width'] * d_outputs['area']\n            if 'width' in d_inputs:\n                d_inputs['width'] += inputs['length'] * d_outputs['area']\n")),(0,s.kt)("a",{id:"csdl.core.custom_implicit_operation"}),(0,s.kt)("h1",{id:"csdlcorecustom_implicit_operation"},"csdl.core.custom","_","implicit","_","operation"),(0,s.kt)("a",{id:"csdl.core.custom_implicit_operation.CustomImplicitOperation"}),(0,s.kt)("h2",{id:"customimplicitoperation-objects"},"CustomImplicitOperation Objects"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class CustomImplicitOperation(CustomOperation)\n")),(0,s.kt)("a",{id:"csdl.core.custom_implicit_operation.CustomImplicitOperation.evaluate_residuals"}),(0,s.kt)("h3",{id:"evaluate_residuals"},"evaluate","_","residuals"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def evaluate_residuals(inputs, outputs, residuals)\n")),(0,s.kt)("p",null,"User defined method to evaluate residuals"),(0,s.kt)("p",null,(0,s.kt)("em",{parentName:"p"},"Example")),(0,s.kt)("p",null,".. code-block:: python"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"def evaluate_residuals(self, inputs, outputs, residuals):\n    x = outputs['x']\n    a = inputs['a']\n    b = inputs['b']\n    c = inputs['c']\n    residuals['x'] = a * x**2 + b * x + c\n")),(0,s.kt)("a",{id:"csdl.core.custom_implicit_operation.CustomImplicitOperation.compute_derivatives"}),(0,s.kt)("h3",{id:"compute_derivatives-1"},"compute","_","derivatives"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def compute_derivatives(inputs, outputs, derivatives)\n")),(0,s.kt)("p",null,"[Optional]"," User defined method to evaluate exact derivatives of\nresiduals wrt inputs and outputs"),(0,s.kt)("p",null,(0,s.kt)("em",{parentName:"p"},"Example")),(0,s.kt)("p",null,".. code-block:: python"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"def compute_derivatives(self, inputs, outputs, derivatives):\n    a = inputs['a']\n    b = inputs['b']\n    x = outputs['x']\n\n    derivatives['x', 'a'] = x**2\n    derivatives['x', 'b'] = x\n    derivatives['x', 'c'] = 1.0\n    derivatives['x', 'x'] = 2 * a * x + b\n\n    # only necessary if implementing `apply_inverse_jacobian`\n    self.inv_jac = 1.0 / (2 * a * x + b)\n")),(0,s.kt)("a",{id:"csdl.core.custom_implicit_operation.CustomImplicitOperation.solve_residual_equations"}),(0,s.kt)("h3",{id:"solve_residual_equations"},"solve","_","residual","_","equations"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def solve_residual_equations(inputs, outputs)\n")),(0,s.kt)("p",null,"[Optional]"," User defined method to solve residual equations,\ncomputing the outputs given the inputs. Define this method to\nimplement a custom solver. Assigning a nonlinear solver will\ncause ",(0,s.kt)("inlineCode",{parentName:"p"},"evaluate_residual_equations")," to run instead."),(0,s.kt)("p",null,(0,s.kt)("em",{parentName:"p"},"Example")),(0,s.kt)("p",null,".. code-block:: python"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"def solve_residual_equations(self, inputs, outputs):\n    a = inputs['a']\n    b = inputs['b']\n    c = inputs['c']\n    outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)\n")),(0,s.kt)("a",{id:"csdl.core.custom_implicit_operation.CustomImplicitOperation.apply_inverse_jacobian"}),(0,s.kt)("h3",{id:"apply_inverse_jacobian"},"apply","_","inverse","_","jacobian"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def apply_inverse_jacobian(d_outputs, d_residuals, mode)\n")),(0,s.kt)("p",null,"[Optional]"," Solve linear system. Invoked when solving coupled\nlinear system; i.e. when solving Newton system to update\nimplicit state variables, and when computing total derivatives"),(0,s.kt)("p",null,(0,s.kt)("em",{parentName:"p"},"Example")),(0,s.kt)("p",null,".. code-block:: python"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"# using self.inv_jac defined in `compute_derivatives` example\ndef apply_inverse_jacobian( self, d_outputs, d_residuals, mode)\n    if mode == 'fwd':\n        d_outputs['x'] = self.inv_jac * d_residuals['x']\n    elif mode == 'rev':\n        d_residuals['x'] = self.inv_jac * d_outputs['x']\n")),(0,s.kt)("a",{id:"csdl.core.custom_implicit_operation.CustomImplicitOperation.compute_jacvec_product"}),(0,s.kt)("h3",{id:"compute_jacvec_product-1"},"compute","_","jacvec","_","product"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"def compute_jacvec_product(inputs, outputs, d_inputs, d_outputs, d_residuals, mode)\n")),(0,s.kt)("p",null,"[Optional]"," Implement partial derivatives by computing a\nmatrix-vector product."),(0,s.kt)("p",null,(0,s.kt)("em",{parentName:"p"},"Example")),(0,s.kt)("p",null,".. code-block:: python"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"def compute_jacvec_product(\n        self,\n        inputs,\n        outputs,\n        d_inputs,\n        d_outputs,\n        d_residuals,\n        mode,\n    ):\n        a = inputs['a']\n        b = inputs['b']\n        c = inputs['c']\n        x = outputs['x']\n        if mode == 'fwd':\n            if 'x' in d_residuals:\n                if 'x' in d_outputs:\n                    d_residuals['x'] += (2 * a * x + b) * d_outputs['x']\n                if 'a' in d_inputs:\n                    d_residuals['x'] += x ** 2 * d_inputs['a']\n                if 'b' in d_inputs:\n                    d_residuals['x'] += x * d_inputs['b']\n                if 'c' in d_inputs:\n                    d_residuals['x'] += d_inputs['c']\n        elif mode == 'rev':\n            if 'x' in d_residuals:\n                if 'x' in d_outputs:\n                    d_outputs['x'] += (2 * a * x + b) * d_residuals['x']\n                if 'a' in d_inputs:\n                    d_inputs['a'] += x ** 2 * d_residuals['x']\n                if 'b' in d_inputs:\n                    d_inputs['b'] += x * d_residuals['x']\n                if 'c' in d_inputs:\n                    d_inputs['c'] += d_residuals['x']\n")))}c.isMDXComponent=!0}}]);