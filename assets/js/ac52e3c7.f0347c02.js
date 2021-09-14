"use strict";(self.webpackChunkmy_website=self.webpackChunkmy_website||[]).push([[2839],{3905:function(n,e,t){t.d(e,{Zo:function(){return l},kt:function(){return f}});var r=t(7294);function i(n,e,t){return e in n?Object.defineProperty(n,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):n[e]=t,n}function o(n,e){var t=Object.keys(n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(n);e&&(r=r.filter((function(e){return Object.getOwnPropertyDescriptor(n,e).enumerable}))),t.push.apply(t,r)}return t}function s(n){for(var e=1;e<arguments.length;e++){var t=null!=arguments[e]?arguments[e]:{};e%2?o(Object(t),!0).forEach((function(e){i(n,e,t[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(n,Object.getOwnPropertyDescriptors(t)):o(Object(t)).forEach((function(e){Object.defineProperty(n,e,Object.getOwnPropertyDescriptor(t,e))}))}return n}function a(n,e){if(null==n)return{};var t,r,i=function(n,e){if(null==n)return{};var t,r,i={},o=Object.keys(n);for(r=0;r<o.length;r++)t=o[r],e.indexOf(t)>=0||(i[t]=n[t]);return i}(n,e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(n);for(r=0;r<o.length;r++)t=o[r],e.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(n,t)&&(i[t]=n[t])}return i}var p=r.createContext({}),c=function(n){var e=r.useContext(p),t=e;return n&&(t="function"==typeof n?n(e):s(s({},e),n)),t},l=function(n){var e=c(n.components);return r.createElement(p.Provider,{value:e},n.children)},m={inlineCode:"code",wrapper:function(n){var e=n.children;return r.createElement(r.Fragment,{},e)}},u=r.forwardRef((function(n,e){var t=n.components,i=n.mdxType,o=n.originalType,p=n.parentName,l=a(n,["components","mdxType","originalType","parentName"]),u=c(t),f=i,d=u["".concat(p,".").concat(f)]||u[f]||m[f]||o;return t?r.createElement(d,s(s({ref:e},l),{},{components:t})):r.createElement(d,s({ref:e},l))}));function f(n,e){var t=arguments,i=e&&e.mdxType;if("string"==typeof n||i){var o=t.length,s=new Array(o);s[0]=u;var a={};for(var p in e)hasOwnProperty.call(e,p)&&(a[p]=e[p]);a.originalType=n,a.mdxType="string"==typeof n?n:i,s[1]=a;for(var c=2;c<o;c++)s[c]=t[c];return r.createElement.apply(null,s)}return r.createElement.apply(null,t)}u.displayName="MDXCreateElement"},7449:function(n,e,t){t.r(e),t.d(e,{frontMatter:function(){return a},contentTitle:function(){return p},metadata:function(){return c},toc:function(){return l},default:function(){return u}});var r=t(7462),i=t(3366),o=(t(7294),t(3905)),s=["components"],a={},p=void 0,c={unversionedId:"worked_examples/ex_min_axiswise",id:"worked_examples/ex_min_axiswise",isDocsHomePage:!1,title:"ex_min_axiswise",description:"`py",source:"@site/docs/worked_examples/ex_min_axiswise.mdx",sourceDirName:"worked_examples",slug:"/worked_examples/ex_min_axiswise",permalink:"/csdl/docs/worked_examples/ex_min_axiswise",editUrl:"https://github.com/lsdolab/csdl/edit/main/website/docs/worked_examples/ex_min_axiswise.mdx",tags:[],version:"current",frontMatter:{}},l=[],m={toc:l};function u(n){var e=n.components,t=(0,i.Z)(n,s);return(0,o.kt)("wrapper",(0,r.Z)({},m,t,{components:e,mdxType:"MDXLayout"}),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-py"},"from csdl_om import Simulator\nfrom csdl import Model\nimport csdl\nimport numpy as np\n\n\nclass ExampleAxiswise(Model):\n    def define(self):\n        m = 2\n        n = 3\n        o = 4\n        p = 5\n        q = 6\n\n        # Shape of a tensor\n        tensor_shape = (m, n, o, p, q)\n\n        num_of_elements = np.prod(tensor_shape)\n        # Creating the values of the tensor\n        val = np.arange(num_of_elements).reshape(tensor_shape)\n\n        # Declaring the tensor as an input\n        ten = self.declare_variable('tensor', val=val)\n\n        # Computing the axiswise minimum on the tensor\n        axis = 1\n        self.register_output('AxiswiseMin', csdl.min(ten, axis=axis))\n\n\nsim = Simulator(ExampleAxiswise())\nsim.run()\n\nprint('tensor', sim['tensor'].shape)\nprint(sim['tensor'])\nprint('AxiswiseMin', sim['AxiswiseMin'].shape)\nprint(sim['AxiswiseMin'])\n")),(0,o.kt)("pre",null,(0,o.kt)("code",{parentName:"pre",className:"language-tensor",metastring:"(2, 3, 4, 5, 6)","(2,":!0,"3,":!0,"4,":!0,"5,":!0,"6)":!0},"[[[[[  0.   1.   2.   3.   4.   5.]\n    [  6.   7.   8.   9.  10.  11.]\n    [ 12.  13.  14.  15.  16.  17.]\n    [ 18.  19.  20.  21.  22.  23.]\n    [ 24.  25.  26.  27.  28.  29.]]\n\n   [[ 30.  31.  32.  33.  34.  35.]\n    [ 36.  37.  38.  39.  40.  41.]\n    [ 42.  43.  44.  45.  46.  47.]\n    [ 48.  49.  50.  51.  52.  53.]\n    [ 54.  55.  56.  57.  58.  59.]]\n\n   [[ 60.  61.  62.  63.  64.  65.]\n    [ 66.  67.  68.  69.  70.  71.]\n    [ 72.  73.  74.  75.  76.  77.]\n    [ 78.  79.  80.  81.  82.  83.]\n    [ 84.  85.  86.  87.  88.  89.]]\n\n   [[ 90.  91.  92.  93.  94.  95.]\n    [ 96.  97.  98.  99. 100. 101.]\n    [102. 103. 104. 105. 106. 107.]\n    [108. 109. 110. 111. 112. 113.]\n    [114. 115. 116. 117. 118. 119.]]]\n\n\n  [[[120. 121. 122. 123. 124. 125.]\n    [126. 127. 128. 129. 130. 131.]\n    [132. 133. 134. 135. 136. 137.]\n    [138. 139. 140. 141. 142. 143.]\n    [144. 145. 146. 147. 148. 149.]]\n\n   [[150. 151. 152. 153. 154. 155.]\n    [156. 157. 158. 159. 160. 161.]\n    [162. 163. 164. 165. 166. 167.]\n    [168. 169. 170. 171. 172. 173.]\n    [174. 175. 176. 177. 178. 179.]]\n\n   [[180. 181. 182. 183. 184. 185.]\n    [186. 187. 188. 189. 190. 191.]\n    [192. 193. 194. 195. 196. 197.]\n    [198. 199. 200. 201. 202. 203.]\n    [204. 205. 206. 207. 208. 209.]]\n\n   [[210. 211. 212. 213. 214. 215.]\n    [216. 217. 218. 219. 220. 221.]\n    [222. 223. 224. 225. 226. 227.]\n    [228. 229. 230. 231. 232. 233.]\n    [234. 235. 236. 237. 238. 239.]]]\n\n\n  [[[240. 241. 242. 243. 244. 245.]\n    [246. 247. 248. 249. 250. 251.]\n    [252. 253. 254. 255. 256. 257.]\n    [258. 259. 260. 261. 262. 263.]\n    [264. 265. 266. 267. 268. 269.]]\n\n   [[270. 271. 272. 273. 274. 275.]\n    [276. 277. 278. 279. 280. 281.]\n    [282. 283. 284. 285. 286. 287.]\n    [288. 289. 290. 291. 292. 293.]\n    [294. 295. 296. 297. 298. 299.]]\n\n   [[300. 301. 302. 303. 304. 305.]\n    [306. 307. 308. 309. 310. 311.]\n    [312. 313. 314. 315. 316. 317.]\n    [318. 319. 320. 321. 322. 323.]\n    [324. 325. 326. 327. 328. 329.]]\n\n   [[330. 331. 332. 333. 334. 335.]\n    [336. 337. 338. 339. 340. 341.]\n    [342. 343. 344. 345. 346. 347.]\n    [348. 349. 350. 351. 352. 353.]\n    [354. 355. 356. 357. 358. 359.]]]]\n\n\n\n [[[[360. 361. 362. 363. 364. 365.]\n    [366. 367. 368. 369. 370. 371.]\n    [372. 373. 374. 375. 376. 377.]\n    [378. 379. 380. 381. 382. 383.]\n    [384. 385. 386. 387. 388. 389.]]\n\n   [[390. 391. 392. 393. 394. 395.]\n    [396. 397. 398. 399. 400. 401.]\n    [402. 403. 404. 405. 406. 407.]\n    [408. 409. 410. 411. 412. 413.]\n    [414. 415. 416. 417. 418. 419.]]\n\n   [[420. 421. 422. 423. 424. 425.]\n    [426. 427. 428. 429. 430. 431.]\n    [432. 433. 434. 435. 436. 437.]\n    [438. 439. 440. 441. 442. 443.]\n    [444. 445. 446. 447. 448. 449.]]\n\n   [[450. 451. 452. 453. 454. 455.]\n    [456. 457. 458. 459. 460. 461.]\n    [462. 463. 464. 465. 466. 467.]\n    [468. 469. 470. 471. 472. 473.]\n    [474. 475. 476. 477. 478. 479.]]]\n\n\n  [[[480. 481. 482. 483. 484. 485.]\n    [486. 487. 488. 489. 490. 491.]\n    [492. 493. 494. 495. 496. 497.]\n    [498. 499. 500. 501. 502. 503.]\n    [504. 505. 506. 507. 508. 509.]]\n\n   [[510. 511. 512. 513. 514. 515.]\n    [516. 517. 518. 519. 520. 521.]\n    [522. 523. 524. 525. 526. 527.]\n    [528. 529. 530. 531. 532. 533.]\n    [534. 535. 536. 537. 538. 539.]]\n\n   [[540. 541. 542. 543. 544. 545.]\n    [546. 547. 548. 549. 550. 551.]\n    [552. 553. 554. 555. 556. 557.]\n    [558. 559. 560. 561. 562. 563.]\n    [564. 565. 566. 567. 568. 569.]]\n\n   [[570. 571. 572. 573. 574. 575.]\n    [576. 577. 578. 579. 580. 581.]\n    [582. 583. 584. 585. 586. 587.]\n    [588. 589. 590. 591. 592. 593.]\n    [594. 595. 596. 597. 598. 599.]]]\n\n\n  [[[600. 601. 602. 603. 604. 605.]\n    [606. 607. 608. 609. 610. 611.]\n    [612. 613. 614. 615. 616. 617.]\n    [618. 619. 620. 621. 622. 623.]\n    [624. 625. 626. 627. 628. 629.]]\n\n   [[630. 631. 632. 633. 634. 635.]\n    [636. 637. 638. 639. 640. 641.]\n    [642. 643. 644. 645. 646. 647.]\n    [648. 649. 650. 651. 652. 653.]\n    [654. 655. 656. 657. 658. 659.]]\n\n   [[660. 661. 662. 663. 664. 665.]\n    [666. 667. 668. 669. 670. 671.]\n    [672. 673. 674. 675. 676. 677.]\n    [678. 679. 680. 681. 682. 683.]\n    [684. 685. 686. 687. 688. 689.]]\n\n   [[690. 691. 692. 693. 694. 695.]\n    [696. 697. 698. 699. 700. 701.]\n    [702. 703. 704. 705. 706. 707.]\n    [708. 709. 710. 711. 712. 713.]\n    [714. 715. 716. 717. 718. 719.]]]]]\nAxiswiseMin (2, 4, 5, 6)\n[[[[  0.   1.   2.   3.   4.   5.]\n   [  6.   7.   8.   9.  10.  11.]\n   [ 12.  13.  14.  15.  16.  17.]\n   [ 18.  19.  20.  21.  22.  23.]\n   [ 24.  25.  26.  27.  28.  29.]]\n\n  [[ 30.  31.  32.  33.  34.  35.]\n   [ 36.  37.  38.  39.  40.  41.]\n   [ 42.  43.  44.  45.  46.  47.]\n   [ 48.  49.  50.  51.  52.  53.]\n   [ 54.  55.  56.  57.  58.  59.]]\n\n  [[ 60.  61.  62.  63.  64.  65.]\n   [ 66.  67.  68.  69.  70.  71.]\n   [ 72.  73.  74.  75.  76.  77.]\n   [ 78.  79.  80.  81.  82.  83.]\n   [ 84.  85.  86.  87.  88.  89.]]\n\n  [[ 90.  91.  92.  93.  94.  95.]\n   [ 96.  97.  98.  99. 100. 101.]\n   [102. 103. 104. 105. 106. 107.]\n   [108. 109. 110. 111. 112. 113.]\n   [114. 115. 116. 117. 118. 119.]]]\n\n\n [[[360. 361. 362. 363. 364. 365.]\n   [366. 367. 368. 369. 370. 371.]\n   [372. 373. 374. 375. 376. 377.]\n   [378. 379. 380. 381. 382. 383.]\n   [384. 385. 386. 387. 388. 389.]]\n\n  [[390. 391. 392. 393. 394. 395.]\n   [396. 397. 398. 399. 400. 401.]\n   [402. 403. 404. 405. 406. 407.]\n   [408. 409. 410. 411. 412. 413.]\n   [414. 415. 416. 417. 418. 419.]]\n\n  [[420. 421. 422. 423. 424. 425.]\n   [426. 427. 428. 429. 430. 431.]\n   [432. 433. 434. 435. 436. 437.]\n   [438. 439. 440. 441. 442. 443.]\n   [444. 445. 446. 447. 448. 449.]]\n\n  [[450. 451. 452. 453. 454. 455.]\n   [456. 457. 458. 459. 460. 461.]\n   [462. 463. 464. 465. 466. 467.]\n   [468. 469. 470. 471. 472. 473.]\n   [474. 475. 476. 477. 478. 479.]]]]\n")))}u.isMDXComponent=!0}}]);