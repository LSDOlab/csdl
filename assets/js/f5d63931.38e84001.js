"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[6509],{3905:function(e,r,t){t.d(r,{Zo:function(){return s},kt:function(){return d}});var n=t(7294);function l(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function i(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);r&&(n=n.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,n)}return t}function a(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?i(Object(t),!0).forEach((function(r){l(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function o(e,r){if(null==e)return{};var t,n,l=function(e,r){if(null==e)return{};var t,n,l={},i=Object.keys(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||(l[t]=e[t]);return l}(e,r);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(n=0;n<i.length;n++)t=i[n],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(l[t]=e[t])}return l}var c=n.createContext({}),p=function(e){var r=n.useContext(c),t=r;return e&&(t="function"==typeof e?e(r):a(a({},r),e)),t},s=function(e){var r=p(e.components);return n.createElement(c.Provider,{value:r},e.children)},u={inlineCode:"code",wrapper:function(e){var r=e.children;return n.createElement(n.Fragment,{},r)}},m=n.forwardRef((function(e,r){var t=e.components,l=e.mdxType,i=e.originalType,c=e.parentName,s=o(e,["components","mdxType","originalType","parentName"]),m=p(t),d=l,f=m["".concat(c,".").concat(d)]||m[d]||u[d]||i;return t?n.createElement(f,a(a({ref:r},s),{},{components:t})):n.createElement(f,a({ref:r},s))}));function d(e,r){var t=arguments,l=r&&r.mdxType;if("string"==typeof e||l){var i=t.length,a=new Array(i);a[0]=m;var o={};for(var c in r)hasOwnProperty.call(r,c)&&(o[c]=r[c]);o.originalType=e,o.mdxType="string"==typeof e?e:l,a[1]=o;for(var p=2;p<i;p++)a[p]=t[p];return n.createElement.apply(null,a)}return n.createElement.apply(null,t)}m.displayName="MDXCreateElement"},9937:function(e,r,t){t.r(r),t.d(r,{frontMatter:function(){return o},contentTitle:function(){return c},metadata:function(){return p},toc:function(){return s},default:function(){return m}});var n=t(7462),l=t(3366),i=(t(7294),t(3905)),a=["components"],o={},c=void 0,p={unversionedId:"worked_examples/ex_implicit_multiple_residuals",id:"worked_examples/ex_implicit_multiple_residuals",isDocsHomePage:!1,title:"ex_implicit_multiple_residuals",description:"`py",source:"@site/docs/worked_examples/ex_implicit_multiple_residuals.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_implicit_multiple_residuals",permalink:"/csdl/docs/worked_examples/ex_implicit_multiple_residuals",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_implicit_multiple_residuals.mdx",tags:[],version:"current",frontMatter:{}},s=[],u={toc:s};function m(e){var r=e.components,t=(0,l.Z)(e,a);return(0,i.kt)("wrapper",(0,n.Z)({},u,t,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model, ScipyKrylov, NewtonSolver, NonlinearBlockGS\nimport numpy as np\n\n\nclass ExampleMultipleResiduals(Model):\n    def define(self):\n        m = Model()\n        r = m.declare_variable('r')\n        a = m.declare_variable('a')\n        b = m.declare_variable('b')\n        c = m.declare_variable('c')\n        x = m.declare_variable('x', val=1.5)\n        y = m.declare_variable('y', val=0.9)\n        m.register_output('rx', x**2 + (y - r)**2 - r**2)\n        m.register_output('ry', a * y**2 + b * y + c)\n\n        r = self.declare_variable('r', val=2)\n        a = self.declare_variable('a', val=1)\n        b = self.declare_variable('b', val=-3)\n        c = self.declare_variable('c', val=2)\n        solve_multiple_implicit = self.create_implicit_operation(m)\n        solve_multiple_implicit.declare_state('x', residual='rx')\n        solve_multiple_implicit.declare_state('y', residual='ry')\n        solve_multiple_implicit.linear_solver = ScipyKrylov()\n        solve_multiple_implicit.nonlinear_solver = NewtonSolver(\n            solve_subsystems=False)\n\n        x, y = solve_multiple_implicit(r, a, b, c)\n\n\nsim = Simulator(ExampleMultipleResiduals())\nsim.run()\n\nprint('x', sim['x'].shape)\nprint(sim['x'])\nprint('y', sim['y'].shape)\nprint(sim['y'])\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-set()"},"[]\n['rx', 'ry']\n\n================\nimplicit_op_0050\n================\nNL: Newton Converged in 4 iterations\nx (1,)\n[1.73205081]\ny (1,)\n[1.]\n")))}m.isMDXComponent=!0}}]);