use crate::genotype_graph::{G, P};
use crate::{Real, U8};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView3, Zip};

#[cfg(feature = "leak-resist-new")]
pub use tp_fixedpoint::timing_shield::{TpEq, TpOrd, TpBool, TpU8};

pub fn viterbi_new(first_tprob_dip: ArrayView1<Real>, tprob_dip: ArrayView3<Real>, genotype_graph: ArrayView1<G>) -> Array1<u8> {
    //TODO fix this
    #[cfg(feature = "leak-resist-new")]
    let m = (genotype_graph.len() + 2) / 3;

    #[cfg(feature = "leak-resist-new")]
    let (tprob_dip, genotype_graph_reduced) = {
        let mut tprob_dip_ = Array3::<Real>::zeros((m, P, P));
        let mut genotype_graph_reduced = vec![TpBool::protect(false); m];
        let mut last_marker_i = None;
        for i in 0..tprob_dip.len() {
            if genotype_graph[i].is_segment_marker() {
                last_marker_i = Some(i);
            }
            if (i % 3 == 2 || i == tprob_dip.len() - 1) && last_marker_i.is_some() {
                tprob_dip_
                    .slice_mut(s![i / 3, .., ..])
                    .assign(&tprob_dip.slice(s![last_marker_i.unwrap(), .., ..]));
                genotype_graph_reduced[i / 3] = TpBool::protect(last_marker_i.is_some());
                last_marker_i = None;
            }
        }
        (tprob_dip_, genotype_graph_reduced)
    };

    #[cfg(not(feature = "leak-resist-new"))]
    let m = genotype_graph.iter().filter(|g| g.is_segment_marker()).count();

    let mut backtrace = Array2::<U8>::zeros((m, P));
    let mut maxprob = first_tprob_dip.to_owned(); 
    let mut maxprob_next = Array1::<Real>::zeros(P);

    let mut max_val: Real = 0.;
    let mut max_ind: U8 = 0;

    for i in 0..m {
        for h2 in 0..P {
            for h1 in 0..P {
                let val = maxprob[h1] * tprob_dip[[i, h1, h2]];
                if h1 == 0 {
                    max_val = val;
                    #[cfg(feature = "leak-resist-new")]
                    {
                        max_ind = TpU8::protect(h1 as u8);
                    }
                    #[cfg(not(feature = "leak-resist-new"))]
                    {
                        max_ind = h1 as u8;
                    }
                } else {
                    #[cfg(feature = "leak-resist")]
                    {
                        let update = val.tp_gt(&max_val);
                        max_val = update.select(val, max_val);
                        max_ind = update.select(TpU8::protect(h1 as u8), max_ind);
                    }
                    #[cfg(not(feature = "leak-resist-new"))]
                    if val > max_val {
                        max_val = val;
                        max_ind = h1 as u8;
                    }
                }
            }

            maxprob_next[h2] = max_val;
            backtrace[[i, h2]] = max_ind;
        }
        maxprob = &maxprob_next / maxprob_next.sum();
    }

    // Find max value in the final vector
    for h1 in 0..P {
        let val = maxprob[h1];
        if h1 == 0 {
            max_val = val;
            #[cfg(feature = "leak-resist-new")]
            {
                max_ind = TpU8::protect(h1 as u8);
            }
            #[cfg(not(feature = "leak-resist-new"))]
            {
                max_ind = h1 as u8;
            }
        } else {
            #[cfg(feature = "leak-resist-new")]
            {
                let update = val.tp_gt(&max_val);
                max_val = update.select(val, max_val);
                max_ind = update.select(TpU8::protect(h1 as u8), max_ind);
            }
            #[cfg(not(feature = "leak-resist-new"))]
            {
                if val > max_val {
                    max_val = val;
                    max_ind = h1 as u8;
                }
            }
        }
    }

    // Follow backtrace pointers to recover optimal sequence
    let mut map_ind = Array1::<u8>::zeros(m+1);
    map_ind[m] = max_ind;
    for i in (0..m).rev() {
        #[cfg(feature = "leak-resist")]
        {
            max_ind = backtrace[[i, max_ind.expose() as usize]]; // TODO: Use linear ORAM!
        }
        #[cfg(not(feature = "leak-resist-new"))]
        {
            max_ind = backtrace[[i, max_ind as usize]]; // TODO: Use linear ORAM!
        }
        map_ind[i] = max_ind;
    }

    map_ind
}

pub fn viterbi(tprob_dip: ArrayView3<Real>, genotype_graph: ArrayView1<G>) -> Array1<u8> {
    //TODO fix this
    #[cfg(feature = "leak-resist-new")]
    let m = (genotype_graph.len() + 2) / 3;

    #[cfg(feature = "leak-resist-new")]
    let (tprob_dip, genotype_graph_reduced) = {
        let mut tprob_dip_ = Array3::<Real>::zeros((m, P, P));
        let mut genotype_graph_reduced = vec![TpBool::protect(false); m];
        let mut last_marker_i = None;
        for i in 0..tprob_dip.len() {
            if genotype_graph[i].is_segment_marker() {
                last_marker_i = Some(i);
            }
            if (i % 3 == 2 || i == tprob_dip.len() - 1) && last_marker_i.is_some() {
                tprob_dip_
                    .slice_mut(s![i / 3, .., ..])
                    .assign(&tprob_dip.slice(s![last_marker_i.unwrap(), .., ..]));
                genotype_graph_reduced[i / 3] = TpBool::protect(last_marker_i.is_some());
                last_marker_i = None;
            }
        }
        (tprob_dip_, genotype_graph_reduced)
    };

    #[cfg(not(feature = "leak-resist-new"))]
    let m = genotype_graph.iter().map(|g| g.is_segment_marker()).count();

    #[cfg(not(feature = "leak-resist-new"))]
    let (first_tprob_dip, tprob_dip) = {
        let mut tprob_dip_ = Array3::<Real>::zeros((m, P, P));
        let iter = tprob_dip
            .outer_iter()
            .into_iter()
            .zip(genotype_graph.iter().map(|g| g.is_segment_marker()))
            .filter_map(|(t, b)| if b { Some(t) } else { None })
            .zip(tprob_dip_.outer_iter_mut().into_iter());
        for (t_src, mut t_tar) in iter {
            t_tar.assign(&t_src);
        }
        (tprob_dip.slice(s![0, 0, ..]), tprob_dip_)
    };

    let mut backtrace = Array2::<U8>::zeros((m, P));
    let mut maxprob = first_tprob_dip.to_owned(); 
    let mut maxprob_next = Array1::<Real>::zeros(P);

    let mut max_val: Real = 0.;
    let mut max_ind: U8 = 0;

    for i in 0..m {
        for h2 in 0..P {
            for h1 in 0..P {
                let val = maxprob[h1] * tprob_dip[[i, h1, h2]];
                if h1 == 0 {
                    max_val = val;
                    #[cfg(feature = "leak-resist-new")]
                    {
                        max_ind = TpU8::protect(h1 as u8);
                    }
                    #[cfg(not(feature = "leak-resist-new"))]
                    {
                        max_ind = h1 as u8;
                    }
                } else {
                    #[cfg(feature = "leak-resist")]
                    {
                        let update = val.tp_gt(&max_val);
                        max_val = update.select(val, max_val);
                        max_ind = update.select(TpU8::protect(h1 as u8), max_ind);
                    }
                    #[cfg(not(feature = "leak-resist-new"))]
                    if val > max_val {
                        max_val = val;
                        max_ind = h1 as u8;
                    }
                }
            }

            maxprob_next[h2] = max_val;
            backtrace[[i, h2]] = max_ind;
        }
        maxprob = &maxprob_next / maxprob_next.sum();
    }

    // Find max value in the final vector
    for h1 in 0..P {
        let val = maxprob[h1];
        if h1 == 0 {
            max_val = val;
            #[cfg(feature = "leak-resist-new")]
            {
                max_ind = TpU8::protect(h1 as u8);
            }
            #[cfg(not(feature = "leak-resist-new"))]
            {
                max_ind = h1 as u8;
            }
        } else {
            #[cfg(feature = "leak-resist-new")]
            {
                let update = val.tp_gt(&max_val);
                max_val = update.select(val, max_val);
                max_ind = update.select(TpU8::protect(h1 as u8), max_ind);
            }
            #[cfg(not(feature = "leak-resist-new"))]
            {
                if val > max_val {
                    max_val = val;
                    max_ind = h1 as u8;
                }
            }
        }
    }

    // Follow backtrace pointers to recover optimal sequence
    let mut map_ind = Array1::<u8>::zeros(m+1);
    map_ind[m] = max_ind;
    for i in (0..m).rev() {
        #[cfg(feature = "leak-resist")]
        {
            max_ind = backtrace[[i, max_ind.expose() as usize]]; // TODO: Use linear ORAM!
        }
        #[cfg(not(feature = "leak-resist-new"))]
        {
            max_ind = backtrace[[i, max_ind as usize]]; // TODO: Use linear ORAM!
        }
        map_ind[i] = max_ind;
    }

    //TODO fix this
    let mut map_ind_ = Array1::<U8>::zeros(genotype_graph.len());
    let mut cur_ind_iter = map_ind.iter().cloned();
    let mut cur_ind = cur_ind_iter.next().unwrap();
    Zip::from(&mut map_ind_)
        .and(&genotype_graph)
        .for_each(|i, g| {
            if g.is_segment_marker() {
                cur_ind = cur_ind_iter.next().unwrap();
            }
            *i = cur_ind;
        });
    let map_ind = map_ind_;

    map_ind
}

//pub fn viterbi(tprob_dips: ArrayView3<f64>, genotype_graph: ArrayView1<G>) -> Array1<u8> {
    //let m = tprob_dips.shape()[0];
    //let p = tprob_dips.shape()[1];

    //let mut backtrace = Array2::<u8>::zeros((m, p));
    //let mut maxprob = Array1::<f64>::zeros(p);
    //let mut maxprob_next = Array1::<f64>::zeros(p);
    //for h1 in 0..p {
        //maxprob[h1] = tprob_dips[[0, 0, h1]]; // p(x1), diploid prob
    //}

    //let mut max_val: f64 = 0.;
    //let mut max_ind: u8 = 0;

    //for i in 1..m {
        //for h2 in 0..p {
            //for h1 in 0..p {
                //let val = if genotype_graph[i].is_segment_marker() {
                    //maxprob[h1] * tprob_dips[[i, h1, h2]]
                //} else {
                    //if h1 == h2 {
                        //maxprob[h1]
                    //} else {
                        //0.
                    //}
                //};
                //if h1 == 0 {
                    //max_val = val;
                    //#[cfg(feature = "leak-resist")]
                    //{
                        //max_ind = U8::protect(h1 as u8);
                    //}
                    //#[cfg(not(feature = "leak-resist"))]
                    //{
                        //max_ind = h1 as u8;
                    //}
                //} else {
                    //#[cfg(feature = "leak-resist")]
                    //{
                        //let update = val.tp_gt(&max_val);
                        //max_val = update.select(val, max_val);
                        //max_ind = update.select(U8::protect(h1 as u8), max_ind);
                    //}
                    //#[cfg(not(feature = "leak-resist"))]
                    //{
                        //if val > max_val {
                            //max_val = val;
                            //max_ind = h1 as u8;
                        //}
                    //}
                //}
            //}

            //maxprob_next[h2] = max_val;
            //backtrace[[i, h2]] = max_ind;
        //}
        //maxprob = &maxprob_next / maxprob_next.sum();
    //}

    //// Find max value in the final vector
    //for h1 in 0..p {
        //let val = maxprob[h1];
        //if h1 == 0 {
            //max_val = val;
            //#[cfg(feature = "leak-resist")]
            //{
                //max_ind = U8::protect(h1 as u8);
            //}
            //#[cfg(not(feature = "leak-resist"))]
            //{
                //max_ind = h1 as u8;
            //}
        //} else {
            //#[cfg(feature = "leak-resist")]
            //{
                //let update = val.tp_gt(&max_val);
                //max_val = update.select(val, max_val);
                //max_ind = update.select(U8::protect(h1 as u8), max_ind);
            //}
            //#[cfg(not(feature = "leak-resist"))]
            //{
                //if val > max_val {
                    //max_val = val;
                    //max_ind = h1 as u8;
                //}
            //}
        //}
    //}

    //// Follow backtrace pointers to recover optimal sequence
    //let mut map_ind = Array1::<u8>::zeros(m);
    //map_ind[m - 1] = max_ind;
    //for i in (1..m).rev() {
        //#[cfg(feature = "leak-resist")]
        //{
            //max_ind = backtrace[[i, max_ind.expose() as usize]]; // TODO: Use linear ORAM!
        //}
        //#[cfg(not(feature = "leak-resist"))]
        //{
            //max_ind = backtrace[[i, max_ind as usize]]; // TODO: Use linear ORAM!
        //}
        //map_ind[i - 1] = max_ind;
    //}
    //map_ind
//}
