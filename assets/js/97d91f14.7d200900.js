"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[6843],{3905:function(e,t,r){r.d(t,{Zo:function(){return p},kt:function(){return _}});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function i(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?i(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):i(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function s(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},i=Object.keys(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)r=i[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var l=n.createContext({}),c=function(e){var t=n.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},p=function(e){var t=c(e.components);return n.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},m=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,i=e.originalType,l=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),m=c(r),_=o,d=m["".concat(l,".").concat(_)]||m[_]||u[_]||i;return r?n.createElement(d,a(a({ref:t},p),{},{components:r})):n.createElement(d,a({ref:t},p))}));function _(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var i=r.length,a=new Array(i);a[0]=m;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:o,a[1]=s;for(var c=2;c<i;c++)a[c]=r[c];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}m.displayName="MDXCreateElement"},2685:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return c},toc:function(){return p},default:function(){return m}});var n=r(7462),o=r(3366),i=(r(7294),r(3905)),a=["components"],s={},l=void 0,c={unversionedId:"worked_examples/ex_implicit_expose_with_subsystems_with_expose",id:"worked_examples/ex_implicit_expose_with_subsystems_with_expose",isDocsHomePage:!1,title:"ex_implicit_expose_with_subsystems_with_expose",description:"`py",source:"@site/docs/worked_examples/ex_implicit_expose_with_subsystems_with_expose.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_implicit_expose_with_subsystems_with_expose",permalink:"/csdl/docs/worked_examples/ex_implicit_expose_with_subsystems_with_expose",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_implicit_expose_with_subsystems_with_expose.mdx",tags:[],version:"current",frontMatter:{}},p=[],u={toc:p};function m(e){var t=e.components,r=(0,o.Z)(e,a);return(0,i.kt)("wrapper",(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS\nimport numpy as np\n\n\nclass ExampleWithSubsystemsWithExpose(Model):\n    def define(self):\n        with self.create_submodel('R') as model:\n            p = model.create_input('p', val=7)\n            q = model.create_input('q', val=8)\n            r = p + q\n            model.register_output('r', r)\n        r = self.declare_variable('r')\n\n        m2 = Model()\n        a = m2.declare_variable('a')\n        r = m2.register_output('r', a - ((a + 3 - a**4) / 2)**(1 / 4))\n        t2 = m2.register_output('t2', a**2)\n\n        m3 = Model()\n        a = m3.declare_variable('a')\n        b = m3.declare_variable('b')\n        c = m3.declare_variable('c')\n        r = m3.declare_variable('r')\n        y = m3.declare_variable('y')\n        m3.register_output('z', a * y**2 + b * y + c - r)\n        m3.register_output('t3', a + b + c - r)\n        m3.register_output('t4', y**2)\n\n        solve_fixed_point_iteration = self.create_implicit_operation(m2)\n        solve_fixed_point_iteration.declare_state('a', residual='r')\n        solve_fixed_point_iteration.nonlinear_solver = NonlinearBlockGS(\n            maxiter=100)\n        a, t2 = solve_fixed_point_iteration(expose=['t2'])\n\n        solve_quadratic = self.create_implicit_operation(m3)\n        b = self.create_input('b', val=-4)\n        solve_quadratic.declare_state('y', residual='z')\n        solve_quadratic.nonlinear_solver = NewtonSolver(\n            solve_subsystems=False,\n            maxiter=100,\n            iprint=False,\n        )\n        solve_quadratic.linear_solver = ScipyKrylov()\n\n        c = self.declare_variable('c', val=18)\n        y, t3, t4 = solve_quadratic(a, b, c, r, expose=['t3', 't4'])\n\n\nsim = Simulator(ExampleWithSubsystemsWithExpose())\nsim.run()\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"================\nimplicit_op_00cN\n================\nNL: NLBGS Converged in 26 iterations\n")))}m.isMDXComponent=!0}}]);