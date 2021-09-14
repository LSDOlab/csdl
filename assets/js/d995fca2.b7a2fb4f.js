"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[3267],{3905:function(e,r,n){n.d(r,{Zo:function(){return l},kt:function(){return f}});var t=n(7294);function o(e,r,n){return r in e?Object.defineProperty(e,r,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[r]=n,e}function c(e,r){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);r&&(t=t.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),n.push.apply(n,t)}return n}function a(e){for(var r=1;r<arguments.length;r++){var n=null!=arguments[r]?arguments[r]:{};r%2?c(Object(n),!0).forEach((function(r){o(e,r,n[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):c(Object(n)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))}))}return e}function i(e,r){if(null==e)return{};var n,t,o=function(e,r){if(null==e)return{};var n,t,o={},c=Object.keys(e);for(t=0;t<c.length;t++)n=c[t],r.indexOf(n)>=0||(o[n]=e[n]);return o}(e,r);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);for(t=0;t<c.length;t++)n=c[t],r.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(o[n]=e[n])}return o}var u=t.createContext({}),s=function(e){var r=t.useContext(u),n=r;return e&&(n="function"==typeof e?e(r):a(a({},r),e)),n},l=function(e){var r=s(e.components);return t.createElement(u.Provider,{value:r},e.children)},p={inlineCode:"code",wrapper:function(e){var r=e.children;return t.createElement(t.Fragment,{},r)}},m=t.forwardRef((function(e,r){var n=e.components,o=e.mdxType,c=e.originalType,u=e.parentName,l=i(e,["components","mdxType","originalType","parentName"]),m=s(n),f=o,d=m["".concat(u,".").concat(f)]||m[f]||p[f]||c;return n?t.createElement(d,a(a({ref:r},l),{},{components:n})):t.createElement(d,a({ref:r},l))}));function f(e,r){var n=arguments,o=r&&r.mdxType;if("string"==typeof e||o){var c=n.length,a=new Array(c);a[0]=m;var i={};for(var u in r)hasOwnProperty.call(r,u)&&(i[u]=r[u]);i.originalType=e,i.mdxType="string"==typeof e?e:o,a[1]=i;for(var s=2;s<c;s++)a[s]=n[s];return t.createElement.apply(null,a)}return t.createElement.apply(null,n)}m.displayName="MDXCreateElement"},9646:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return i},contentTitle:function(){return u},metadata:function(){return s},toc:function(){return l},default:function(){return m}});var t=n(7462),o=n(3366),c=(n(7294),n(3905)),a=["components"],i={},u=void 0,s={unversionedId:"worked_examples/ex_einsum_new_outer_vector_vector",id:"worked_examples/ex_einsum_new_outer_vector_vector",isDocsHomePage:!1,title:"ex_einsum_new_outer_vector_vector",description:"`py",source:"@site/docs/worked_examples/ex_einsum_new_outer_vector_vector.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_new_outer_vector_vector",permalink:"/csdl/docs/worked_examples/ex_einsum_new_outer_vector_vector",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_new_outer_vector_vector.mdx",tags:[],version:"current",frontMatter:{}},l=[],p={toc:l};function m(e){var r=e.components,n=(0,o.Z)(e,a);return(0,c.kt)("wrapper",(0,t.Z)({},p,n,{components:r,mdxType:"MDXLayout"}),(0,c.kt)("pre",null,(0,c.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleOuterVectorVector(Model):\n    def define(self):\n        a = np.arange(4)\n        vec = self.declare_variable('a', val=a)\n\n        self.register_output(\n            'einsum_outer1',\n            csdl.einsum_new_api(\n                vec,\n                vec,\n                operation=[('rows', ), ('cols', ), ('rows', 'cols')],\n            ))\n\n\nsim = Simulator(ExampleOuterVectorVector())\nsim.run()\n\nprint('a', sim['a'].shape)\nprint(sim['a'])\nprint('einsum_outer1', sim['einsum_outer1'].shape)\nprint(sim['einsum_outer1'])\n")),(0,c.kt)("pre",null,(0,c.kt)("code",{parentName:"pre",className:"language-a",metastring:"(4,)","(4,)":!0},"[0. 1. 2. 3.]\neinsum_outer1 (4, 4)\n[[0. 0. 0. 0.]\n [0. 1. 2. 3.]\n [0. 2. 4. 6.]\n [0. 3. 6. 9.]]\n")))}m.isMDXComponent=!0}}]);