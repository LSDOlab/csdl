"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[6824,2263],{3905:function(e,t,r){r.d(t,{Zo:function(){return l},kt:function(){return m}});var n=r(7294);function o(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function c(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function a(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?c(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):c(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function u(e,t){if(null==e)return{};var r,n,o=function(e,t){if(null==e)return{};var r,n,o={},c=Object.keys(e);for(n=0;n<c.length;n++)r=c[n],t.indexOf(r)>=0||(o[r]=e[r]);return o}(e,t);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);for(n=0;n<c.length;n++)r=c[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var i=n.createContext({}),s=function(e){var t=n.useContext(i),r=t;return e&&(r="function"==typeof e?e(t):a(a({},t),e)),r},l=function(e){var t=s(e.components);return n.createElement(i.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},d=n.forwardRef((function(e,t){var r=e.components,o=e.mdxType,c=e.originalType,i=e.parentName,l=u(e,["components","mdxType","originalType","parentName"]),d=s(r),m=o,v=d["".concat(i,".").concat(m)]||d[m]||p[m]||c;return r?n.createElement(v,a(a({ref:t},l),{},{components:r})):n.createElement(v,a({ref:t},l))}));function m(e,t){var r=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var c=r.length,a=new Array(c);a[0]=d;var u={};for(var i in t)hasOwnProperty.call(t,i)&&(u[i]=t[i]);u.originalType=e,u.mdxType="string"==typeof e?e:o,a[1]=u;for(var s=2;s<c;s++)a[s]=r[s];return n.createElement.apply(null,a)}return n.createElement.apply(null,r)}d.displayName="MDXCreateElement"},9036:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return i},contentTitle:function(){return s},metadata:function(){return l},toc:function(){return p},default:function(){return m}});var n=r(7462),o=r(3366),c=(r(7294),r(3905)),a=r(4061),u=["components"],i={},s="Outer Product between Two Vectors",l={unversionedId:"examples/Standard Library/outer/ex_outer_vector_vector",id:"examples/Standard Library/outer/ex_outer_vector_vector",isDocsHomePage:!1,title:"Outer Product between Two Vectors",description:"This is an example of how to use the csdl outer function to compute",source:"@site/docs/examples/Standard Library/outer/ex_outer_vector_vector.mdx",sourceDirName:"examples/Standard Library/outer",slug:"/examples/Standard Library/outer/ex_outer_vector_vector",permalink:"/csdl/docs/examples/Standard Library/outer/ex_outer_vector_vector",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/outer/ex_outer_vector_vector.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Outer Product between a Tensor and a Vector",permalink:"/csdl/docs/examples/Standard Library/outer/ex_outer_tensor_vector"},next:{title:"Axis Free Pnorm Computation",permalink:"/csdl/docs/examples/Standard Library/pnorm/ex_pnorm_axisfree"}},p=[],d={toc:p};function m(e){var t=e.components,r=(0,o.Z)(e,u);return(0,c.kt)("wrapper",(0,n.Z)({},d,r,{components:t,mdxType:"MDXLayout"}),(0,c.kt)("h1",{id:"outer-product-between-two-vectors"},"Outer Product between Two Vectors"),(0,c.kt)("p",null,"This is an example of how to use the csdl outer function to compute\nthe outer product between two vectors."),(0,c.kt)(a.default,{mdxType:"WorkedExample"}))}m.isMDXComponent=!0},4061:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return u},contentTitle:function(){return i},metadata:function(){return s},toc:function(){return l},default:function(){return d}});var n=r(7462),o=r(3366),c=(r(7294),r(3905)),a=["components"],u={},i=void 0,s={unversionedId:"worked_examples/ex_outer_vector_vector",id:"worked_examples/ex_outer_vector_vector",isDocsHomePage:!1,title:"ex_outer_vector_vector",description:"`py",source:"@site/docs/worked_examples/ex_outer_vector_vector.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_outer_vector_vector",permalink:"/csdl/docs/worked_examples/ex_outer_vector_vector",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_outer_vector_vector.mdx",tags:[],version:"current",frontMatter:{}},l=[],p={toc:l};function d(e){var t=e.components,r=(0,o.Z)(e,a);return(0,c.kt)("wrapper",(0,n.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,c.kt)("pre",null,(0,c.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleVectorVector(Model):\n    def define(self):\n\n        m = 3\n\n        # Shape of the vectors\n        vec_shape = (m, )\n\n        # Values for the two vectors\n        vec1 = np.arange(m)\n        vec2 = np.arange(m, 2 * m)\n\n        # Adding the vectors to csdl\n        vec1 = self.declare_variable('vec1', val=vec1)\n        vec2 = self.declare_variable('vec2', val=vec2)\n\n        # Vector-Vector Outer Product\n        self.register_output('VecVecOuter', csdl.outer(vec1, vec2))\n\n\nsim = Simulator(ExampleVectorVector())\nsim.run()\n\nprint('vec1', sim['vec1'].shape)\nprint(sim['vec1'])\nprint('vec2', sim['vec2'].shape)\nprint(sim['vec2'])\nprint('VecVecOuter', sim['VecVecOuter'].shape)\nprint(sim['VecVecOuter'])\n")),(0,c.kt)("pre",null,(0,c.kt)("code",{parentName:"pre",className:"language-vec1",metastring:"(3,)","(3,)":!0},"[0. 1. 2.]\nvec2 (3,)\n[3. 4. 5.]\nVecVecOuter (3, 3)\n[[ 0.  0.  0.]\n [ 3.  4.  5.]\n [ 6.  8. 10.]]\n")))}d.isMDXComponent=!0}}]);