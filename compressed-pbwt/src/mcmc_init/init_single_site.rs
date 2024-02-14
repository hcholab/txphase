use super::{InitRank, RankList};

const INIT_NEIGHBORS: usize = 2;
use crate::{Bool, Isize, Usize, U16};

#[cfg(feature = "obliv")]
use timing_shield::{TpEq, TpOrd};

pub fn init_single_site(
    start_site_i: usize,
    site_i: usize,
    id_0: U16,
    d_a_0: U16,
    d_b_0: U16,
    id_1: U16,
    d_a_1: U16,
    d_b_1: U16,
    div_level: &[u16],
    init_level_0: &[RankList<InitRank>],
    init_level_1: &[RankList<InitRank>],
) -> (Bool, Bool) {
    let nn_0 = process_top_init_neighbors(
        start_site_i,
        site_i,
        id_0,
        d_a_0,
        d_b_0,
        div_level,
        init_level_0,
    );
    let nn_1 = process_top_init_neighbors(
        start_site_i,
        site_i,
        id_1,
        d_a_1,
        d_b_1,
        div_level,
        init_level_1,
    );
    init_single_site_from_ranks(&nn_0, &nn_1)
}

pub fn process_top_init_neighbors(
    start_site_i: usize,
    site_i: usize,
    id: U16,
    d_a: U16,
    d_b: U16,
    div_level: &[u16],
    init_level: &[RankList<InitRank>],
) -> Vec<(Usize, Bool)> {
    let mut nn = Vec::new();

    #[cfg(feature = "obliv")]
    let nearest_neighbor = d_a
        .tp_lt_eq(&d_b)
        .select(id.tp_eq(&0).select(id, id - 1), id);

    #[cfg(not(feature = "obliv"))]
    let nearest_neighbor = if d_a <= d_b {
        if id == 0 {
            id
        } else {
            id - 1
        }
    } else {
        id
    };

    #[cfg(feature = "obliv")]
    let (d, d_is_above) = {
        let d_is_above = d_a.tp_lt_eq(&d_b);
        (d_is_above.select(d_a, d_b), d_is_above)
    };

    #[cfg(not(feature = "obliv"))]
    let (d, d_is_above) = if d_a <= d_b {
        (d_a, true)
    } else {
        (d_b, false)
    };

    let (rank, opt_div) = find_top_init_neighbors_level(nearest_neighbor, div_level, init_level);

    nn.push((
        get_match_length(start_site_i, site_i, rank.first().unwrap().get_div(), d),
        rank.first().unwrap().get_hap(),
    ));

    #[cfg(feature = "obliv")]
    let d = {
        let (opt, opt_div, is_above) = opt_div;

        let d_1 = d_is_above.select(
            d_a.tp_gt(&opt_div).select(d_a, opt_div),
            d_b.tp_gt(&opt_div).select(d_b, opt_div),
        );

        let d_2 = is_above.select(d_a, d_b);
        let d_3 = d_is_above.tp_eq(&is_above).select(d_1, d_2);
        opt.select(d_3, d)
    };

    #[cfg(not(feature = "obliv"))]
    let d = if let Some((opt_div, is_above)) = opt_div {
        if d_is_above == is_above {
            if d_is_above {
                d_a.max(opt_div)
            } else {
                d_b.max(opt_div)
            }
        } else {
            if is_above {
                d_a
            } else {
                d_b
            }
        }
    } else {
        d
    };

    nn.push((
        get_match_length(start_site_i, site_i, rank.last().unwrap().get_div(), d),
        rank.last().unwrap().get_hap(),
    ));
    nn
}

fn init_single_site_from_ranks(nn_0: &[(Usize, Bool)], nn_1: &[(Usize, Bool)]) -> (Bool, Bool) {
    #[cfg(feature = "obliv")]
    {
        let s = score(nn_0) - score(nn_1);
        let cond1 = s.tp_gt(&0);
        let s_abs = cond1.select(s, -s);
        let cond2 = s_abs.tp_gt_eq(&2);
        let cond3 = check_score_div(nn_0, nn_1);
        let h_0 = (cond2 & cond1) | (!cond2 & cond3);
        let h_1 = !h_0;
        (h_0, h_1)
    }

    #[cfg(not(feature = "obliv"))]
    {
        let s = score(nn_0) - score(nn_1);

        if s.abs() >= 2 {
            if s > 0 {
                (true, false)
            } else {
                (false, true)
            }
        } else {
            if check_score_div(nn_0, nn_1) {
                (true, false)
            } else {
                (false, true)
            }
        }
    }
}

fn score(nn: &[(Usize, Bool)]) -> Isize {
    #[cfg(feature = "obliv")]
    return nn
        .iter()
        .map(|v| v.1.as_i64() * 2 - 1)
        .reduce(|accu, v| accu + v)
        .unwrap();

    #[cfg(not(feature = "obliv"))]
    return nn.iter().map(|v| (v.1 as isize * 2 - 1)).sum();
}

#[cfg(feature = "obliv")]
fn check_score_div(nn_0: &[(Usize, Bool)], nn_1: &[(Usize, Bool)]) -> Bool {
    let score_div_0 = nn_0
        .iter()
        .map(|v| v.1.select((v.0 + 1) * (v.0 + 1), Usize::protect(1)))
        .reduce(|acc, x| acc * x)
        .unwrap()
        * nn_1
            .iter()
            .map(|v| (v.0 + 1))
            .reduce(|acc, x| acc * x)
            .unwrap();
    let score_div_1 = nn_1
        .iter()
        .map(|v| v.1.select((v.0 + 1) * (v.0 + 1), Usize::protect(1)))
        .reduce(|acc, x| acc * x)
        .unwrap()
        * nn_0
            .iter()
            .map(|v| (v.0 + 1))
            .reduce(|acc, x| acc * x)
            .unwrap();
    score_div_0.tp_gt(&score_div_1)
}

#[cfg(not(feature = "obliv"))]
fn check_score_div(nn_0: &[(usize, bool)], nn_1: &[(usize, bool)]) -> bool {
    score_div(nn_0) > score_div(nn_1)
}

#[cfg(not(feature = "obliv"))]
fn score_div(nn: &[(usize, bool)]) -> f64 {
    nn.iter()
        .map(|v| (v.1 as isize * 2 - 1) as f64 * ((v.0 + 1) as f64).ln())
        .sum()
}

fn get_match_length(start_site_i: usize, site_i: usize, prev_div: Usize, rel_div: U16) -> Usize {
    #[cfg(feature = "obliv")]
    let site_i = site_i as u64;

    site_i + 1 - get_abs_div(start_site_i, prev_div, rel_div)
}

fn get_abs_div(start_site_i: usize, prev_div: Usize, rel_div: U16) -> Usize {
    #[cfg(feature = "obliv")]
    return rel_div
        .tp_eq(&0)
        .select(prev_div, start_site_i as u64 + rel_div.as_u64());

    #[cfg(not(feature = "obliv"))]
    if rel_div == 0 {
        prev_div
    } else {
        start_site_i + rel_div as usize
    }
}

#[cfg(feature = "obliv")]
fn find_top_init_neighbors_level(
    input_pos: U16,
    div_level: &[u16],
    init_level: &[RankList<InitRank>],
) -> (RankList<InitRank>, (Bool, U16, Bool)) {
    let mut opt = (Bool::protect(false), U16::protect(0), Bool::protect(false));
    let mut result_ranks = RankList::with_elem(INIT_NEIGHBORS, InitRank::default());
    for i in 0..init_level.len() as u16 {
        let (ranks, opt_div) = find_top_init_neighbors_level_(i, div_level, init_level);
        let cond = input_pos.tp_eq(&i);
        result_ranks.cond_copy_from(&ranks, cond);
        if let Some(opt_div) = opt_div {
            opt.0 = cond;
            opt.1 = cond.select(U16::protect(opt_div.0), opt.1);
            opt.2 = cond.select(Bool::protect(opt_div.1), opt.2);
        }
    }
    (result_ranks, opt)
}

#[cfg(not(feature = "obliv"))]
#[inline]
fn find_top_init_neighbors_level(
    input_pos: u16,
    div_level: &[u16],
    init_level: &[RankList<InitRank>],
) -> (RankList<InitRank>, Option<(u16, bool)>) {
    find_top_init_neighbors_level_(input_pos, div_level, init_level)
}

fn find_top_init_neighbors_level_(
    input_pos: u16,
    div_level: &[u16],
    init_level: &[RankList<InitRank>],
) -> (RankList<InitRank>, Option<(u16, bool)>) {
    let pos = input_pos as usize;
    let mut neighbors = init_level[pos].clone();
    if neighbors.len() == INIT_NEIGHBORS {
        return (neighbors, None);
    }
    let pos_up = if pos == 0 { None } else { Some(pos - 1) };
    let pos_down = if pos + 1 >= div_level.len() {
        None
    } else {
        Some(pos + 1)
    };

    let (new_pos, div, is_above) = match (pos_up, pos_down) {
        (Some(pos_up_), Some(pos_down_)) => {
            let div_up = div_level[pos_up_ + 1];
            let div_down = div_level[pos_down_];
            if div_up <= div_down {
                (pos_up_, div_up, true)
            } else {
                (pos_down_, div_down, false)
            }
        }
        (Some(pos_up_), None) => (pos_up_, div_level[pos_up_ + 1], true),
        (None, Some(pos_down_)) => (pos_down_, div_level[pos_down_], false),
        _ => panic!("This shouldn't happen."),
    };
    neighbors.push(init_level[new_pos].first().unwrap().clone());
    (neighbors, Some((div, is_above)))
}
