"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[4460,9478],{3905:function(e,t,r){r.d(t,{Zo:function(){return m},kt:function(){return d}});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function s(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},o=Object.keys(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(n=0;n<o.length;n++)r=o[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var p=n.createContext({}),c=function(e){var t=n.useContext(p),r=t;return e&&(r="function"==typeof e?e(t):s(s({},t),e)),r},m=function(e){var t=c(e.components);return n.createElement(p.Provider,{value:t},e.children)},l={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},u=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,o=e.originalType,p=e.parentName,m=i(e,["components","mdxType","originalType","parentName"]),u=c(r),d=a,f=u["".concat(p,".").concat(d)]||u[d]||l[d]||o;return r?n.createElement(f,s(s({ref:t},m),{},{components:r})):n.createElement(f,s({ref:t},m))}));function d(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=r.length,s=new Array(o);s[0]=u;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i.mdxType="string"==typeof e?e:a,s[1]=i;for(var c=2;c<o;c++)s[c]=r[c];return n.createElement.apply(null,s)}return n.createElement.apply(null,r)}u.displayName="MDXCreateElement"},5289:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return p},contentTitle:function(){return c},metadata:function(){return m},toc:function(){return l},default:function(){return d}});var n=r(7462),a=r(3366),o=(r(7294),r(3905)),s=r(9834),i=["components"],p={},c="Matrix Transpose",m={unversionedId:"examples/Standard Library/transpose/ex_transpose_matrix",id:"examples/Standard Library/transpose/ex_transpose_matrix",isDocsHomePage:!1,title:"Matrix Transpose",description:"This is an example of taking the transpose of a matrix.",source:"@site/docs/examples/Standard Library/transpose/ex_transpose_matrix.mdx",sourceDirName:"examples/Standard Library/transpose",slug:"/examples/Standard Library/transpose/ex_transpose_matrix",permalink:"/csdl/docs/examples/Standard Library/transpose/ex_transpose_matrix",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/transpose/ex_transpose_matrix.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Sum of a Single Vector",permalink:"/csdl/docs/examples/Standard Library/sum/ex_sum_single_vector"},next:{title:"Tensor Transpose",permalink:"/csdl/docs/examples/Standard Library/transpose/ex_transpose_tensor"}},l=[],u={toc:l};function d(e){var t=e.components,r=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("h1",{id:"matrix-transpose"},"Matrix Transpose"),(0,o.kt)("p",null,"This is an example of taking the transpose of a matrix."),(0,o.kt)(s.default,{mdxType:"WorkedExample"}))}d.isMDXComponent=!0},9834:function(e,t,r){r.r(t),r.d(t,{frontMatter:function(){return i},contentTitle:function(){return p},metadata:function(){return c},toc:function(){return m},default:function(){return u}});var n=r(7462),a=r(3366),o=(r(7294),r(3905)),s=["components"],i={},p=void 0,c={unversionedId:"worked_examples/ex_transpose_matrix",id:"worked_examples/ex_transpose_matrix",isDocsHomePage:!1,title:"ex_transpose_matrix",description:"`py",source:"@site/docs/worked_examples/ex_transpose_matrix.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_transpose_matrix",permalink:"/csdl/docs/worked_examples/ex_transpose_matrix",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_transpose_matrix.mdx",tags:[],version:"current",frontMatter:{}},m=[],l={toc:m};function u(e){var t=e.components,r=(0,a.Z)(e,s);return(0,o.kt)("wrapper",(0,n.Z)({},l,r,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleMatrix(Model):\n    def define(self):\n\n        # Declare mat as an input matrix with shape = (4, 2)\n        mat = self.declare_variable(\n            'Mat',\n            val=np.arange(4 * 2).reshape((4, 2)),\n        )\n\n        # Compute the transpose of mat\n        self.register_output('matrix_transpose', csdl.transpose(mat))\n\n\nsim = Simulator(ExampleMatrix())\nsim.run()\n\nprint('Mat', sim['Mat'].shape)\nprint(sim['Mat'])\nprint('matrix_transpose', sim['matrix_transpose'].shape)\nprint(sim['matrix_transpose'])\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-Mat",metastring:"(4, 2)","(4,":!0,"2)":!0},"[[0. 1.]\n [2. 3.]\n [4. 5.]\n [6. 7.]]\nmatrix_transpose (2, 4)\n[[0. 2. 4. 6.]\n [1. 3. 5. 7.]]\n")))}u.isMDXComponent=!0}}]);