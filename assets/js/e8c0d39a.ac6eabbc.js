"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[3193,6005],{3905:function(e,r,n){n.d(r,{Zo:function(){return c},kt:function(){return l}});var t=n(7294);function i(e,r,n){return r in e?Object.defineProperty(e,r,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[r]=n,e}function a(e,r){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var t=Object.getOwnPropertySymbols(e);r&&(t=t.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),n.push.apply(n,t)}return n}function o(e){for(var r=1;r<arguments.length;r++){var n=null!=arguments[r]?arguments[r]:{};r%2?a(Object(n),!0).forEach((function(r){i(e,r,n[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))}))}return e}function s(e,r){if(null==e)return{};var n,t,i=function(e,r){if(null==e)return{};var n,t,i={},a=Object.keys(e);for(t=0;t<a.length;t++)n=a[t],r.indexOf(n)>=0||(i[n]=e[n]);return i}(e,r);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(t=0;t<a.length;t++)n=a[t],r.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var m=t.createContext({}),u=function(e){var r=t.useContext(m),n=r;return e&&(n="function"==typeof e?e(r):o(o({},r),e)),n},c=function(e){var r=u(e.components);return t.createElement(m.Provider,{value:r},e.children)},p={inlineCode:"code",wrapper:function(e){var r=e.children;return t.createElement(t.Fragment,{},r)}},d=t.forwardRef((function(e,r){var n=e.components,i=e.mdxType,a=e.originalType,m=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),d=u(n),l=i,_=d["".concat(m,".").concat(l)]||d[l]||p[l]||a;return n?t.createElement(_,o(o({ref:r},c),{},{components:n})):t.createElement(_,o({ref:r},c))}));function l(e,r){var n=arguments,i=r&&r.mdxType;if("string"==typeof e||i){var a=n.length,o=new Array(a);o[0]=d;var s={};for(var m in r)hasOwnProperty.call(r,m)&&(s[m]=r[m]);s.originalType=e,s.mdxType="string"==typeof e?e:i,o[1]=s;for(var u=2;u<a;u++)o[u]=n[u];return t.createElement.apply(null,o)}return t.createElement.apply(null,n)}d.displayName="MDXCreateElement"},9096:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return m},contentTitle:function(){return u},metadata:function(){return c},toc:function(){return p},default:function(){return l}});var t=n(7462),i=n(3366),a=(n(7294),n(3905)),o=n(2897),s=["components"],m={},u="Reordering a Matrix using Einsum",c={unversionedId:"examples/Standard Library/einsum_new/ex_einsum_new_reorder_matrix",id:"examples/Standard Library/einsum_new/ex_einsum_new_reorder_matrix",isDocsHomePage:!1,title:"Reordering a Matrix using Einsum",description:"This is an example of how to properly use the einsum function",source:"@site/docs/examples/Standard Library/einsum_new/ex_einsum_new_reorder_matrix.mdx",sourceDirName:"examples/Standard Library/einsum_new",slug:"/examples/Standard Library/einsum_new/ex_einsum_new_reorder_matrix",permalink:"/csdl/docs/examples/Standard Library/einsum_new/ex_einsum_new_reorder_matrix",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/einsum_new/ex_einsum_new_reorder_matrix.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Vector-Vector Outer Product using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_new/ex_einsum_new_outer_vector_vector_sparse"},next:{title:"Reordering a Matrix using Einsum with Sparse Partials",permalink:"/csdl/docs/examples/Standard Library/einsum_new/ex_einsum_new_reorder_matrix_sparse"}},p=[],d={toc:p};function l(e){var r=e.components,n=(0,i.Z)(e,s);return(0,a.kt)("wrapper",(0,t.Z)({},d,n,{components:r,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"reordering-a-matrix-using-einsum"},"Reordering a Matrix using Einsum"),(0,a.kt)("p",null,"This is an example of how to properly use the einsum function\nto reorder a matrix."),(0,a.kt)(o.default,{mdxType:"WorkedExample"}))}l.isMDXComponent=!0},2897:function(e,r,n){n.r(r),n.d(r,{frontMatter:function(){return s},contentTitle:function(){return m},metadata:function(){return u},toc:function(){return c},default:function(){return d}});var t=n(7462),i=n(3366),a=(n(7294),n(3905)),o=["components"],s={},m=void 0,u={unversionedId:"worked_examples/ex_einsum_new_reorder_matrix",id:"worked_examples/ex_einsum_new_reorder_matrix",isDocsHomePage:!1,title:"ex_einsum_new_reorder_matrix",description:"`py",source:"@site/docs/worked_examples/ex_einsum_new_reorder_matrix.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_einsum_new_reorder_matrix",permalink:"/csdl/docs/worked_examples/ex_einsum_new_reorder_matrix",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_einsum_new_reorder_matrix.mdx",tags:[],version:"current",frontMatter:{}},c=[],p={toc:c};function d(e){var r=e.components,n=(0,i.Z)(e,o);return(0,a.kt)("wrapper",(0,t.Z)({},p,n,{components:r,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nfrom csdl import Model\nimport csdl\n\n\nclass ExampleReorderMatrix(Model):\n    def define(self):\n        shape2 = (5, 4)\n        b = np.arange(20).reshape(shape2)\n        mat = self.declare_variable('b', val=b)\n\n        # reorder of a matrix\n        self.register_output(\n            'einsum_reorder1',\n            csdl.einsum_new_api(mat, operation=[(46, 99), (99, 46)]))\n\n\nsim = Simulator(ExampleReorderMatrix())\nsim.run()\n\nprint('b', sim['b'].shape)\nprint(sim['b'])\nprint('einsum_reorder1', sim['einsum_reorder1'].shape)\nprint(sim['einsum_reorder1'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-b",metastring:"(5, 4)","(5,":!0,"4)":!0},"[[ 0.  1.  2.  3.]\n [ 4.  5.  6.  7.]\n [ 8.  9. 10. 11.]\n [12. 13. 14. 15.]\n [16. 17. 18. 19.]]\neinsum_reorder1 (4, 5)\n[[ 0.  4.  8. 12. 16.]\n [ 1.  5.  9. 13. 17.]\n [ 2.  6. 10. 14. 18.]\n [ 3.  7. 11. 15. 19.]]\n")))}d.isMDXComponent=!0}}]);