"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[2254,5550,888,9580],{3905:function(e,n,t){t.d(n,{Zo:function(){return c},kt:function(){return u}});var r=t(7294);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function a(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function s(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?a(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):a(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,r,i=function(e,n){if(null==e)return{};var t,r,i={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(i[t]=e[t]);return i}(e,n);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var l=r.createContext({}),p=function(e){var n=r.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):s(s({},n),e)),t},c=function(e){var n=p(e.components);return r.createElement(l.Provider,{value:n},e.children)},d={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,i=e.mdxType,a=e.originalType,l=e.parentName,c=o(e,["components","mdxType","originalType","parentName"]),m=p(t),u=i,x=m["".concat(l,".").concat(u)]||m[u]||d[u]||a;return t?r.createElement(x,s(s({ref:n},c),{},{components:t})):r.createElement(x,s({ref:n},c))}));function u(e,n){var t=arguments,i=n&&n.mdxType;if("string"==typeof e||i){var a=t.length,s=new Array(a);s[0]=m;var o={};for(var l in n)hasOwnProperty.call(n,l)&&(o[l]=n[l]);o.originalType=e,o.mdxType="string"==typeof e?e:i,s[1]=o;for(var p=2;p<a;p++)s[p]=t[p];return r.createElement.apply(null,s)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},2665:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return c},contentTitle:function(){return d},metadata:function(){return m},toc:function(){return u},default:function(){return f}});var r=t(7462),i=t(3366),a=(t(7294),t(3905)),s=t(3593),o=t(998),l=t(6237),p=["components"],c={},d="Array Indexing",m={unversionedId:"examples/Basic Examples/indexing",id:"examples/Basic Examples/indexing",isDocsHomePage:!1,title:"Array Indexing",description:"`csdl supports indexing into Variable` objects for explicit",source:"@site/docs/examples/Basic Examples/indexing.mdx",sourceDirName:"examples/Basic Examples",slug:"/examples/Basic Examples/indexing",permalink:"/csdl/docs/examples/Basic Examples/indexing",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/examples/Basic Examples/indexing.mdx",tags:[],version:"current",frontMatter:{},sidebar:"docs",previous:{title:"Implicit Relationships",permalink:"/csdl/docs/examples/Basic Examples/implicit_relationships"},next:{title:"Input to Main Model",permalink:"/csdl/docs/examples/Basic Examples/input"}},u=[],x={toc:u};function f(e){var n=e.components,t=(0,i.Z)(e,p);return(0,a.kt)("wrapper",(0,r.Z)({},x,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"array-indexing"},"Array Indexing"),(0,a.kt)("p",null,(0,a.kt)("inlineCode",{parentName:"p"},"csdl")," supports indexing into ",(0,a.kt)("inlineCode",{parentName:"p"},"Variable")," objects for explicit\noutputs."),(0,a.kt)("p",null,"In this example, integer indices are used to concatenate multiple\nexpressions/variables into one variable and extract values from a single\nvariable representing an array."),(0,a.kt)("p",null,"The variable representing the array, ",(0,a.kt)("inlineCode",{parentName:"p"},"'x'"),", is created with the\n",(0,a.kt)("inlineCode",{parentName:"p"},"Model.create_output")," method.\nArray index assignments are then used to define ",(0,a.kt)("inlineCode",{parentName:"p"},"'x'")," in terms of\nother CSDL variables."),(0,a.kt)("p",null,"Note: For every variable created with ",(0,a.kt)("inlineCode",{parentName:"p"},"Model.create_output"),", there is no\nneed to call ",(0,a.kt)("inlineCode",{parentName:"p"},"Model.register_output"),"."),(0,a.kt)(s.default,{mdxType:"WorkedExample1"}),(0,a.kt)("p",null,(0,a.kt)("inlineCode",{parentName:"p"},"csdl")," supports specifying ranges as well as individual indices to\nslice and concatenate arrays."),(0,a.kt)(o.default,{mdxType:"WorkedExample2"}),(0,a.kt)("p",null,(0,a.kt)("inlineCode",{parentName:"p"},"csdl")," supports specifying ranges along multiple axes as well as\nindividual indices and ranges to slice and concatenate arrays."),(0,a.kt)(l.default,{mdxType:"WorkedExample3"}))}f.isMDXComponent=!0},3593:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return o},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return c},default:function(){return m}});var r=t(7462),i=t(3366),a=(t(7294),t(3905)),s=["components"],o={},l=void 0,p={unversionedId:"worked_examples/ex_indices_integer",id:"worked_examples/ex_indices_integer",isDocsHomePage:!1,title:"ex_indices_integer",description:"`py",source:"@site/docs/worked_examples/ex_indices_integer.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_indices_integer",permalink:"/csdl/docs/worked_examples/ex_indices_integer",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_indices_integer.mdx",tags:[],version:"current",frontMatter:{}},c=[],d={toc:c};function m(e){var n=e.components,t=(0,i.Z)(e,s);return(0,a.kt)("wrapper",(0,r.Z)({},d,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nimport csdl\nfrom csdl import Model\n\n\nclass ExampleInteger(Model):\n    def define(self):\n        a = self.declare_variable('a', val=0)\n        b = self.declare_variable('b', val=1)\n        c = self.declare_variable('c', val=2)\n        d = self.declare_variable('d', val=7.4)\n        e = self.declare_variable('e', val=np.pi)\n        f = self.declare_variable('f', val=9)\n        g = e + f\n        x = self.create_output('x', shape=(7, ))\n        x[0] = a\n        x[1] = b\n        x[2] = c\n        x[3] = d\n        x[4] = e\n        x[5] = f\n        x[6] = g\n\n        # Get value from indices\n        self.register_output('x0', x[0])\n        self.register_output('x6', x[6])\n        self.register_output('x_2', x[-2])\n\n\nsim = Simulator(ExampleInteger())\nsim.run()\n\nprint('x', sim['x'].shape)\nprint(sim['x'])\nprint('x0', sim['x0'].shape)\nprint(sim['x0'])\nprint('x6', sim['x6'].shape)\nprint(sim['x6'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-x",metastring:"(7,)","(7,)":!0},"[ 0.          1.          2.          7.4         3.14159265  9.\n 12.14159265]\nx0 (1,)\n[0.]\nx6 (1,)\n[12.14159265]\n")))}m.isMDXComponent=!0},6237:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return o},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return c},default:function(){return m}});var r=t(7462),i=t(3366),a=(t(7294),t(3905)),s=["components"],o={},l=void 0,p={unversionedId:"worked_examples/ex_indices_multidimensional",id:"worked_examples/ex_indices_multidimensional",isDocsHomePage:!1,title:"ex_indices_multidimensional",description:"`py",source:"@site/docs/worked_examples/ex_indices_multidimensional.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_indices_multidimensional",permalink:"/csdl/docs/worked_examples/ex_indices_multidimensional",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_indices_multidimensional.mdx",tags:[],version:"current",frontMatter:{}},c=[],d={toc:c};function m(e){var n=e.components,t=(0,i.Z)(e,s);return(0,a.kt)("wrapper",(0,r.Z)({},d,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nimport csdl\nfrom csdl import Model\n\n\nclass ExampleMultidimensional(Model):\n    def define(self):\n        # Works with two dimensional arrays\n        z = self.declare_variable('z',\n                                  shape=(2, 3),\n                                  val=np.arange(6).reshape((2, 3)))\n        x = self.create_output('x', shape=(2, 3))\n        x[0:2, 0:3] = z\n\n        # Also works with higher dimensional arrays\n        p = self.declare_variable('p',\n                                  shape=(5, 2, 3),\n                                  val=np.arange(30).reshape((5, 2, 3)))\n        q = self.create_output('q', shape=(5, 2, 3))\n        q[0:5, 0:2, 0:3] = p\n\n        # Get value from indices\n        self.register_output('r', p[0, :, :])\n\n        # Assign a vector to a slice\n        vec = self.create_input(\n            'vec',\n            shape=(1, 20),\n            val=np.arange(20).reshape((1, 20)),\n        )\n        s = self.create_output('s', shape=(2, 20))\n        s[0, :] = vec\n        s[1, :] = 2 * vec\n\n        # Negative indices\n        t = self.create_output('t', shape=(5, 3, 3), val=0)\n        t[0:5, 0:-1, 0:3] = p\n\n\nsim = Simulator(ExampleMultidimensional())\nsim.run()\n\nprint('x', sim['x'].shape)\nprint(sim['x'])\nprint('q', sim['q'].shape)\nprint(sim['q'])\nprint('r', sim['r'].shape)\nprint(sim['r'])\nprint('s', sim['s'].shape)\nprint(sim['s'])\nprint('t', sim['t'].shape)\nprint(sim['t'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-x",metastring:"(2, 3)","(2,":!0,"3)":!0},"[[0. 1. 2.]\n [3. 4. 5.]]\nq (5, 2, 3)\n[[[ 0.  1.  2.]\n  [ 3.  4.  5.]]\n\n [[ 6.  7.  8.]\n  [ 9. 10. 11.]]\n\n [[12. 13. 14.]\n  [15. 16. 17.]]\n\n [[18. 19. 20.]\n  [21. 22. 23.]]\n\n [[24. 25. 26.]\n  [27. 28. 29.]]]\nr (1, 2, 3)\n[[[0. 1. 2.]\n  [3. 4. 5.]]]\ns (2, 20)\n[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.\n  18. 19.]\n [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34.\n  36. 38.]]\nt (5, 3, 3)\n[[[ 0.  1.  2.]\n  [ 3.  4.  5.]\n  [ 1.  1.  1.]]\n\n [[ 6.  7.  8.]\n  [ 9. 10. 11.]\n  [ 1.  1.  1.]]\n\n [[12. 13. 14.]\n  [15. 16. 17.]\n  [ 1.  1.  1.]]\n\n [[18. 19. 20.]\n  [21. 22. 23.]\n  [ 1.  1.  1.]]\n\n [[24. 25. 26.]\n  [27. 28. 29.]\n  [ 1.  1.  1.]]]\n")))}m.isMDXComponent=!0},998:function(e,n,t){t.r(n),t.d(n,{frontMatter:function(){return o},contentTitle:function(){return l},metadata:function(){return p},toc:function(){return c},default:function(){return m}});var r=t(7462),i=t(3366),a=(t(7294),t(3905)),s=["components"],o={},l=void 0,p={unversionedId:"worked_examples/ex_indices_one_dimensional",id:"worked_examples/ex_indices_one_dimensional",isDocsHomePage:!1,title:"ex_indices_one_dimensional",description:"`py",source:"@site/docs/worked_examples/ex_indices_one_dimensional.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_indices_one_dimensional",permalink:"/csdl/docs/worked_examples/ex_indices_one_dimensional",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_indices_one_dimensional.mdx",tags:[],version:"current",frontMatter:{}},c=[],d={toc:c};function m(e){var n=e.components,t=(0,i.Z)(e,s);return(0,a.kt)("wrapper",(0,r.Z)({},d,t,{components:n,mdxType:"MDXLayout"}),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nimport numpy as np\nimport csdl\nfrom csdl import Model\n\n\nclass ExampleOneDimensional(Model):\n    def define(self):\n        n = 20\n        u = self.declare_variable('u',\n                                  shape=(n, ),\n                                  val=np.arange(n).reshape((n, )))\n        v = self.declare_variable('v',\n                                  shape=(n - 4, ),\n                                  val=np.arange(n - 4).reshape(\n                                      (n - 4, )))\n        w = self.declare_variable('w',\n                                  shape=(4, ),\n                                  val=16 + np.arange(4).reshape((4, )))\n        x = self.create_output('x', shape=(n, ))\n        x[0:n] = 2 * (u + 1)\n        y = self.create_output('y', shape=(n, ))\n        y[0:n - 4] = 2 * (v + 1)\n        y[n - 4:n] = w - 3\n\n        # Get value from indices\n        z = self.create_output('z', shape=(3, ))\n        z[0:3] = csdl.expand(x[2], (3, ))\n        self.register_output('x0_5', x[0:5])\n        self.register_output('x3_', x[3:])\n        self.register_output('x2_4', x[2:4])\n        self.register_output('x_last', x[-1])\n\n\nsim = Simulator(ExampleOneDimensional())\nsim.run()\n\nprint('x', sim['x'].shape)\nprint(sim['x'])\nprint('y', sim['y'].shape)\nprint(sim['y'])\nprint('z', sim['z'].shape)\nprint(sim['z'])\nprint('x0_5', sim['x0_5'].shape)\nprint(sim['x0_5'])\nprint('x3_', sim['x3_'].shape)\nprint(sim['x3_'])\nprint('x2_4', sim['x2_4'].shape)\nprint(sim['x2_4'])\n")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-x",metastring:"(20,)","(20,)":!0},"[ 2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34. 36.\n 38. 40.]\ny (20,)\n[ 2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 13. 14.\n 15. 16.]\nz (3,)\n[6. 6. 6.]\nx0_5 (5,)\n[ 2.  4.  6.  8. 10.]\nx3_ (17,)\n[ 8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34. 36. 38. 40.]\nx2_4 (2,)\n[6. 8.]\n")))}m.isMDXComponent=!0}}]);