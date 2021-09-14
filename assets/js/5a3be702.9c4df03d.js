"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[2426,7752],{3905:function(e,r,t){t.d(r,{Zo:function(){return p},kt:function(){return g}});var a=t(7294);function n(e,r,t){return r in e?Object.defineProperty(e,r,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[r]=t,e}function i(e,r){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);r&&(a=a.filter((function(r){return Object.getOwnPropertyDescriptor(e,r).enumerable}))),t.push.apply(t,a)}return t}function o(e){for(var r=1;r<arguments.length;r++){var t=null!=arguments[r]?arguments[r]:{};r%2?i(Object(t),!0).forEach((function(r){n(e,r,t[r])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):i(Object(t)).forEach((function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(t,r))}))}return e}function l(e,r){if(null==e)return{};var t,a,n=function(e,r){if(null==e)return{};var t,a,n={},i=Object.keys(e);for(a=0;a<i.length;a++)t=i[a],r.indexOf(t)>=0||(n[t]=e[t]);return n}(e,r);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)t=i[a],r.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(n[t]=e[t])}return n}var s=a.createContext({}),m=function(e){var r=a.useContext(s),t=r;return e&&(t="function"==typeof e?e(r):o(o({},r),e)),t},p=function(e){var r=m(e.components);return a.createElement(s.Provider,{value:r},e.children)},u={inlineCode:"code",wrapper:function(e){var r=e.children;return a.createElement(a.Fragment,{},r)}},c=a.forwardRef((function(e,r){var t=e.components,n=e.mdxType,i=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),c=m(t),g=n,d=c["".concat(s,".").concat(g)]||c[g]||u[g]||i;return t?a.createElement(d,o(o({ref:r},p),{},{components:t})):a.createElement(d,o({ref:r},p))}));function g(e,r){var t=arguments,n=r&&r.mdxType;if("string"==typeof e||n){var i=t.length,o=new Array(i);o[0]=c;var l={};for(var s in r)hasOwnProperty.call(r,s)&&(l[s]=r[s]);l.originalType=e,l.mdxType="string"==typeof e?e:n,o[1]=l;for(var m=2;m<i;m++)o[m]=t[m];return a.createElement.apply(null,o)}return a.createElement.apply(null,t)}c.displayName="MDXCreateElement"},2424:function(e,r,t){t.r(r),t.d(r,{frontMatter:function(){return s},contentTitle:function(){return m},metadata:function(){return p},toc:function(){return u},default:function(){return g}});var a=t(7462),n=t(3366),i=(t(7294),t(3905)),o=t(6504),l=["components"],s={},m="Average of Multiple Matrices along Rows",p={unversionedId:"examples/Standard Library/average/ex_average_multiple_matrix_along1",id:"examples/Standard Library/average/ex_average_multiple_matrix_along1",isDocsHomePage:!1,title:"Average of Multiple Matrices along Rows",description:"This is an example of computing the elementwise average of the axiswise average",source:"@site/docs/examples/Standard Library/average/ex_average_multiple_matrix_along1.mdx",sourceDirName:"examples/Standard Library/average",slug:"/examples/Standard Library/average/ex_average_multiple_matrix_along1",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_multiple_matrix_along1",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Standard Library/average/ex_average_multiple_matrix_along1.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Average of Multiple Matrices along Columns",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_multiple_matrix_along0"},next:{title:"Average of Multiple Tensors",permalink:"/csdl/docs/examples/Standard Library/average/ex_average_multiple_tensor"}},u=[],c={toc:u};function g(e){var r=e.components,t=(0,n.Z)(e,l);return(0,i.kt)("wrapper",(0,a.Z)({},c,t,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"average-of-multiple-matrices-along-rows"},"Average of Multiple Matrices along Rows"),(0,i.kt)("p",null,"This is an example of computing the elementwise average of the axiswise average\nof matrices M1 ad M2 along the rows."),(0,i.kt)(o.default,{mdxType:"WorkedExample"}))}g.isMDXComponent=!0},6504:function(e,r,t){t.r(r),t.d(r,{frontMatter:function(){return l},contentTitle:function(){return s},metadata:function(){return m},toc:function(){return p},default:function(){return c}});var a=t(7462),n=t(3366),i=(t(7294),t(3905)),o=["components"],l={},s=void 0,m={unversionedId:"worked_examples/ex_average_multiple_matrix_along1",id:"worked_examples/ex_average_multiple_matrix_along1",isDocsHomePage:!1,title:"ex_average_multiple_matrix_along1",description:"`py",source:"@site/docs/worked_examples/ex_average_multiple_matrix_along1.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_average_multiple_matrix_along1",permalink:"/csdl/docs/worked_examples/ex_average_multiple_matrix_along1",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_average_multiple_matrix_along1.mdx",tags:[],version:"current",frontMatter:{}},p=[],u={toc:p};function c(e){var r=e.components,t=(0,n.Z)(e,o);return(0,i.kt)("wrapper",(0,a.Z)({},u,t,{components:r,mdxType:"MDXLayout"}),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleMultipleMatrixAlong1(Model):\n    def define(self):\n        n = 3\n        m = 6\n\n        # Declare a matrix of shape 3x6 as input\n        M1 = self.declare_variable('M1',\n                                   val=np.arange(n * m).reshape((n, m)))\n\n        # Declare another matrix of shape 3x6 as input\n        M2 = self.declare_variable('M2',\n                                   val=np.arange(n * m,\n                                                 2 * n * m).reshape(\n                                                     (n, m)))\n\n        # Output the elementwise average of the axiswise average of\n        # matrices M1 ad M2 along the columns\n        self.register_output(\n            'multiple_matrix_average_along_1',\n            csdl.average(M1, M2, axes=(1, )),\n        )\n\n\nsim = Simulator(ExampleMultipleMatrixAlong1())\nsim.run()\n\nprint('M1', sim['M1'].shape)\nprint(sim['M1'])\nprint('M2', sim['M2'].shape)\nprint(sim['M2'])\nprint('multiple_matrix_average_along_1',\n      sim['multiple_matrix_average_along_1'].shape)\nprint(sim['multiple_matrix_average_along_1'])\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-M1",metastring:"(3, 6)","(3,":!0,"6)":!0},"[[ 0.  1.  2.  3.  4.  5.]\n [ 6.  7.  8.  9. 10. 11.]\n [12. 13. 14. 15. 16. 17.]]\nM2 (3, 6)\n[[18. 19. 20. 21. 22. 23.]\n [24. 25. 26. 27. 28. 29.]\n [30. 31. 32. 33. 34. 35.]]\nmultiple_matrix_average_along_1 (3,)\n[11.5 17.5 23.5]\n")))}c.isMDXComponent=!0}}]);